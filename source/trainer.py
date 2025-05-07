import os, sys, wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.utils import *
from torch.utils.data import DataLoader
from dataset import Rxrx1

class Norm_Trainer():
    def __init__(self, net, device, config, loss_func, collate):
        self.net = net.to(device)
        self.device = device
        self.config = config
        self.loss_func = loss_func
        self.collate = collate
        self.gen = torch.Generator().manual_seed(56)

    def load_checkpoint(self):
        checkpoint = load_weights(self.config['load_checkpoint'], self.net, self.device)
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        last_epoch = checkpoint['epoch']
        training_loss_values = checkpoint['training_loss_values']
        validation_loss_values = checkpoint['validation_loss_values']
        self.config['batch_size'] = checkpoint['batch_size']
        return (last_epoch, training_loss_values, validation_loss_values)

    def init_wandb(self):
        assert wandb.api.api_key, "the api key has not been set!\n"
        # print(f"wandb key: {wandb.api.api_key}")
        wandb.login(verify=True)
        wandb.init(
            project=self.config['project_name'],
            name=self.config['run_name'],
            config=self.config
        )
    
    def train(self, losser: callable):
        #============= Preparing loaders and LR scheduler... ==================
        self.init_wandb()

        workers = self.config["workers"]
        device = self.device

        train_dataset = Rxrx1(self.config['dataset_dir'],
                        metadata_path=self.config['metadata_path'],
                        subset=self.config["cell_type"], split="train", 
                        channels=self.config['channels'])
        
        val_dataset = Rxrx1(self.config['dataset_dir'],
                        metadata_path=self.config['metadata_path'],
                        subset=self.config["cell_type"], split="val",
                        channels=self.config['channels'])


        train_dataloader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True,
                                      num_workers=workers, drop_last=True, collate_fn=self.collate, prefetch_factor=4, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config["batch_size"], shuffle=True,
                                    num_workers=workers, drop_last=True, collate_fn=self.collate, prefetch_factor=4, pin_memory=True)
        self.opt, self.scheduler = load_opt(self.config, self.net, train_dataloader)
        #============= Loading full checkpoint or backbone + head ==================
        if self.config['load_checkpoint'] is not None:
            assert self.config['backbone_weights'] == None, "Config conflict: can't load a checkpoint and backbone weights."
            assert self.config['head_weights'] == None, "Config conflict: can't load a checkpoint and head weights."
            print('Loading latest checkpoint... ')
            last_epoch, training_loss_values, validation_loss_values = self.load_checkpoint()
            print(f"Checkpoint {self.config['load_checkpoint']} Loaded")
        else:
            last_epoch = 0
            training_loss_values = []  # store every training loss value
            validation_loss_values = []  # store every validation loss value

        if self.config['backbone_weights'] is not None:
            print("Loading backbone weights...")
            self.net.load_backbone_weights(self.config, self.device)
            if self.config['freeze_backbone']:
                self.net.freeze_backbone()
            else:
                print("Warning, you loaded backbone weights without freezing them.")

        if self.config['head_weights'] is not None:
            print("Loading head weights...")
            self.net.load_head_weights(self.config, self.device)

        #============= Training Loop ==================
        global_step = 0
        print("Starting training...", flush=True)
        for epoch in range(last_epoch, int(self.config['epochs'])):
            pbar = tqdm(total=len(train_dataloader), desc=f"Epoch-{epoch}")
            self.net.train()
            for i, (x_batch, siRNA_batch, metadata) in enumerate(train_dataloader):
                global_step += 1
                loss, _ = losser(device, (x_batch, siRNA_batch, metadata), self.net, self.loss_func)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                training_loss_values.append(loss.item())
                wandb.log({"train_loss": loss.item()}, step=global_step)
                # ================= Scheduler step (1 per iteration) =================
                self.scheduler.step()
                wandb.log({"lr":float(self.scheduler.get_lr()[0])}, step=global_step)
                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})

            save_model(os.path.join(self.config['checkpoint_dir'], "checkpoint.pth"), epoch, self.net, self.opt, training_loss_values, validation_loss_values, 
                       self.config['batch_size'], self.config['opt'], self.scheduler)
            if (epoch + 1) % int(self.config['model_save_freq']) == 0:
                name = os.path.join(self.config['checkpoint_dir'], "checkpoint{}".format(epoch + 1))
                save_model(name, epoch, self.net, self.opt, training_loss_values, validation_loss_values,
                           self.config['batch_size'], self.config['opt'], self.scheduler)

            # ================= Validation Loop (after evaluation_freq epochs) ==================
            if (epoch + 1) % int(self.config['evaluation_freq']) == 0:
                print(f"Running Validation-{str(epoch+1)}...")
                val_loss, accuracy = validation(net = self.net,
                                                val_loader = val_dataloader, 
                                                device = self.device,
                                                loss_func = self.loss_func, 
                                                losser = losser, 
                                                epoch = epoch)
                validation_loss_values += val_loss
                mean_loss = sum(val_loss) / len(val_loss) if val_loss else 0  # Division by zero paranoia
                wandb.log({"val_loss": mean_loss}, step=global_step)
                if self.config["log_accuracy"]: wandb.log({"accuracy": accuracy}, step=global_step)
        return training_loss_values, validation_loss_values
