import os
import time
import torch
import torch.nn as nn
from torchmetrics import MeanMetric
from model import Model
from utils import (
    set_seed, load_train_data, load_val_data,
    calc_loss, warmup_scheduler, evaluate
)
from config import Config


def train_step(x, det_true, cate_true, model, optimizer, loss_meter, cls_loss_fn, reg_loss_fn):
    optimizer.zero_grad()
    cls, det, _ = model(x, training=True)
    loss1, loss2 = calc_loss(cate_true, det_true, cls, det, cls_loss_fn, reg_loss_fn)
    total_loss = loss1 + loss2
    total_loss.backward()
    optimizer.step()
    loss_meter.update(total_loss.item())


def main():
    set_seed(42)

    train_paths = {
        'signal': './train/signal.mat',
        'det': './train/label_box.mat',
        'cate': './train/label_cate.mat'
    }
    val_paths = {
        'signal': './val/signal.mat',
        'det': './val/label_box.mat',
        'cate': './val/label_cate.mat'
    }

    train_loader, inp_vocab_size, vocab_size, num_class = load_train_data(
        train_paths['signal'], train_paths['det'], train_paths['cate']    )
    
    val_loader, val_det_labels, val_cate_labels = load_val_data(
        val_paths['signal'], val_paths['det'], val_paths['cate']    )

    EPOCHS = 200
    batch_size = 256
    warm_steps = 2500
    learning_rate = 1e-3
    num_layers = 3
    model_dim = 128
    dff = 256
    drop_rate = 0.25
    p_size = 32
    overlap = 12
    ckpt_dir = './result/model'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    max_len_en = train_loader.dataset.tensors[0].shape[1]
    max_len = train_loader.dataset.tensors[1].shape[1]
    n_patch = int((1000 - cfg.p_size) / (cfg.p_size - cfg.overlap) + 1)
    total_steps = cfg.EPOCHS * len(train_loader)

    model = Model(
        cfg.num_layers, cfg.model_dim, cfg.dff,
        num_class, max_len_en, max_len,
        cfg.p_size, cfg.overlap, n_patch,
        cfg.batch_size, cfg.drop_rate
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.learning_rate, 
        betas=(0.9, 0.98), 
        eps=1e-9
    )
    scheduler = warmup_scheduler(optimizer, cfg.warm_steps, total_steps)

    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()
    train_loss = MeanMetric().to(device)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    best_acc = 0

    ckpt_path = os.path.join(cfg.ckpt_dir, 'latest.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print("Loaded latest checkpoint")

    for epoch in range(cfg.EPOCHS):
        start = time.time()
        train_loss.reset()
        model.train()
        
        for batch, (x, det_true, cate_true) in enumerate(train_loader):
            x = x.to(device)
            det_true = det_true.to(device)
            cate_true = cate_true.to(device)
            train_step(x, det_true, cate_true, model, optimizer, train_loss, cls_loss_fn, reg_loss_fn)
            scheduler.step()
        
        model.eval()
        with torch.no_grad():
            det_pred, cate_pred, pd, _, _ = evaluate(
                val_loader, vocab_size, max_len, model, device
            )
            acc = calc_acc(torch.tensor(cate_pred), val_cate_labels)

        print(f'Epoch {epoch + 1}, Validation Accuracy: {acc:.4f}, PD: {pd:.4f}')
        print(f"Epoch {epoch + 1} training time: {time.time() - start:.2f} seconds")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()
            }, os.path.join(cfg.ckpt_dir, 'best.pth'))
            print(f'Epoch {epoch + 1}, saved best model.')

        if epoch % 25 == 0:
            torch.save({
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()
            }, os.path.join(cfg.ckpt_dir, f'epoch_{epoch + 1}.pth'))
            print(f'Epoch {epoch + 1}, saved checkpoint.')
        torch.save({
            'model': model.state_dict(), 
            'optimizer': optimizer.state_dict()
        }, ckpt_path)


if __name__ == '__main__':
    main()