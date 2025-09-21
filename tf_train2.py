import os
import time
import torch
from EEGART.tf_opt import NoamOpt
from EEGART.tf_loss import LabelSmoothing, SimpleLossCompute
from EEGART.tf_model import create_model
from IO import create_dataset
from preprocess import collate_eeg_batch_channel
from EEGART.tf_utils import save_checkpoint, draw


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()

class model_train_parameter():
    def __init__(self, loss, save, data, train_len):
        self.model = 1 # cumbersome_model UNet_family change to UNET++
        self.block_num = 1
        self.max_epochs = 60
        self.num_workers = 4
        self.batch_size = 32 #30
        self.sample_rate = 256
        self.step_loss = 100  # Decrease learning rate after how many epochs.
        self.milestones = [50, 100, 125, 140]
        self.train_len = train_len
        self.loss = loss
        self.lr = 0.01  # 'Initial learning rate'
        self.save = save
        self.savedata = data
        self.savedir = self.save + '/modelsave'  # directory to save the results
        self.savefig = self.save + '/result'
        self.resume = True  # Use this flag to load last checkpoint for training
        self.resumeLoc = self.save + '/modelsave/checkpoint.pth.tar'
        self.classes = 2  # No of classes in the dataset.
        self.logFile = 'model_trainValLog.txt'  # File that stores the training and validation logs
        self.onGPU = True  # Run on CPU or GPU. If TRUE, then GPU.
        self.pretrained = ''  # Pretrained model
        self.loadpickle = './'

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    device = next(model.parameters()).device
    for i, batch in enumerate(data_iter):
        # move batch to model device
        if hasattr(batch, 'to'):
            batch = batch.to(device)
        #print("run_epoch1:", batch.src.shape)
        #print("run_epoch2:", batch.trg.shape)
        #draw(0, batch.src)
        #draw(1, batch.trg)
        #print("run_epoch3:", batch.src_mask.shape)
        #print("run_epoch3.1:", batch.src_mask)
        #draw(0, batch.src_mask)
        #print("run_epoch4:", batch.trg_mask.shape)
        #draw(3, batch.trg_mask)
        #batch = batch.cuda()
        ## device to cuda
        # --- input: noise, output: target ---
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        # --- input: noise, output: noise ---
        #out = model.forward(batch.src, batch.src[:,:,1:], batch.src_mask, batch.trg_mask)
        # --- input: noise, output: 111.. ---
        #out = model.forward(batch.src, batch.ys, batch.src_mask, batch.trg_mask)
        #print("run_epoch3:", out.shape)
        #loss = loss_compute(out, batch.trg_y, batch.ntokens)
        loss = loss_compute(out, batch.trg, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        #if i % 50 == 1:
        elapsed = time.time() - start
        print("Epoch Step: [%d] Loss: %f Time per Epoch: %f" % (i, loss / batch.ntokens, elapsed))
        start = time.time()
        tokens = 0
    return total_loss / total_tokens


if __name__ == '__main__':
    # --------------0. Hyperparameters---------------
    name = "0909"
    save = './' + name + '_RealEEG'
    data = "./" + name + "_simulate_data/"
    args = model_train_parameter(0, save, data, 39200)#39200, 32200
    start_epoch = 0
    # --------------1. Model Define--------------------------------
    V = 30
    # model = make_model(V, V, N=2)
    model = create_model(V, V, N=2)
    # --------------2. Criterion Define----------------------------
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    # --------------3. Optimizer Define----------------------------
    model_opt = NoamOpt(model.src_embed[1].d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9))
    # --------------4. Dataset Define------------------------------
    # Use config-driven factory to select real or synthetic dataset per split
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    trainset = create_dataset(config_path=config_path, mode='train', seed=42)
    valset = create_dataset(config_path=config_path, mode='val', seed=42)
    testset = create_dataset(config_path=config_path, mode='test', seed=42)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_eeg_batch_channel,
    )

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_eeg_batch_channel,
    )

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_eeg_batch_channel,
    )

    # --------------5. Pre-setting-----------------------------------
    # print('Total network parameters: ' + str(model_opt.model_size))
    if args.onGPU:
        model = model.cuda()
        criterion = criterion.cuda()
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    if args.resume:  # 當機回復
        if os.path.isfile(args.resumeLoc):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resumeLoc, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # --------------6. Create Logger------------------------------
    logFileLoc = args.savedir + '/' + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        # logger.write("Parameters: %s" % (str(model_opt.model_size)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'Loss(Ts)', 'Learning_rate', "Time"))
    logger.flush()

    # --------------7. Run Epoch-----------------------------------
    for epoch in range(start_epoch, args.max_epochs):
        start_time = time.time()
        print("Epoch No.: %d/%d" % (epoch, args.max_epochs))
        # --------------a. training--------------------------------
        model.train()
        # model.forward already applies the reconstructor; use identity generator
        trainloss = run_epoch(
            train_loader,
            model,
            SimpleLossCompute(torch.nn.Identity(), criterion, model_opt),
        )
        print("%d training finish %.4f" % (epoch, trainloss))
        # --------------b. validation & Testing--------------------
        model.eval()
        validloss = run_epoch(
            val_loader,
            model,
            SimpleLossCompute(torch.nn.Identity(), criterion, model_opt),
        )
        testloss = run_epoch(
            test_loader,
            model,
            SimpleLossCompute(torch.nn.Identity(), criterion, model_opt),
        )
        print("%d testing finish %.4f, %.4f" % (epoch, validloss, testloss))
        # --------------c. writing log-----------------------------
        logger.write("\n%d\t%.6f\t\t%.6f\t\t%.6f\t\t%.4f\t\t%.2f" % (epoch, trainloss, validloss, testloss, model_opt.rate(), time.time()-start_time))
        logger.flush()
        # --------------d. save checkpoint-----------------------------
        state = {'epoch': epoch + 1,
                'arch': str(model),
                'state_dict': model.state_dict(),
                'lossTr': trainloss,
                'lossTs': testloss,
                'lossVal': validloss,
                ' lr': model_opt.rate()}
        save_checkpoint(state, args.savedir)
    logger.close()
