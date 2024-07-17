from typing import Dict, Optional, Sequence, Union
import torch
import numpy as np
from torch.utils.data import DataLoader
from MyLosses import AccuracyMeter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

import time

def train_net(
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    trainloader : DataLoader,
    valloader : DataLoader,
    testloader : DataLoader,
    loss : torch.nn.modules.loss._Loss,
    num_epochs : int,
    device : torch.device,
    accuracy_meter : Optional[AccuracyMeter]=None,
    verbose : bool=True,
    time_perf : bool =False,
    track_grads : bool =False
) -> Dict :
    r"""Training function for the ``model`` following ``optimizer`` relatively to ``loss`` on dataset provided by ``trainloader`` for ``num_epochs`` on ``device``.
    Care that ``model`` should already be loaded on ``device``.
    ``valloader`` is used to track accuracy on unseen data all along the training, ``testloader`` is used only once at the end.
    ``accuracy_meter`` is an object that should be such that ``accuracy_meter(output, batch_truth)`` return the number of positive outcomes (not the accuracy).
    ``accuracy_meter`` should also have a ``zero`` method in order to initatilize the accuracy and stack it along the batches. Stacking is done through ``+`` operator. Once all the batches have been passed, the stacked value is divided through ``/`` operator.
    We recommand using floats, numpy arrays, torch tensors and alikes for ``accuracy_meter`` return type.
    ``verbose`` just give regular prints of loss results.
    ``time_perf`` set to ``True`` allows to register the time the model take for one epoch through ``trainloader``
    ``track_grads`` allows to register the mean and max gradient for each variable of the model, and this for every batch and epoch.
    """

    with torch.no_grad():

        trainlosses=[]
        vallosses=[]

        if accuracy_meter is not None:
            trainacc=[]
            valacc=[]

        if time_perf:
            times=[]

        result_dict={}

        if track_grads:
            grad_dict={"batches": len(trainloader), "epochs": num_epochs, "layers": []}
            for n,p in model.named_parameters():
                if p.requires_grad:
                    grad_dict["layers"].append(n)

        #step needed for precomputing on dataset
        if hasattr(loss, "initialize"):
            loss.initialize(model, trainloader)

        for epoch in range(num_epochs):

            #initializing centers at each epoch
            if hasattr(loss, "epoch_update"):
                loss.epoch_update()

            if epoch==(num_epochs//3) or epoch==(2*num_epochs//3):
                for p in optimizer.param_groups:
                    p['lr'] /= 10

            if time_perf:
                start=time.perf_counter()

            #training
            with torch.enable_grad():
                model.train()
                if accuracy_meter is not None:
                    temp_acc=accuracy_meter.zero()
                for batch_id, (batch_data, batch_truth) in enumerate(trainloader):
                    temp_losses=[]
                    out=model(batch_data.to(device))
                    optimizer.zero_grad()
                    loss_value=loss(out,batch_truth.to(device))
                    if accuracy_meter is not None:
                        temp_acc += accuracy_meter(out,batch_truth.to(device))
                    temp_losses.append(loss_value.detach().cpu().numpy())
                    loss_value.backward()
                    if track_grads:
                        ave_grads = []
                        max_grads= []
                        for _, p in model.named_parameters():
                            if p.requires_grad:
                                ave_grads.append(p.grad.abs().mean().cpu())
                                max_grads.append(p.grad.abs().max().cpu())
                        grad_dict[epoch, batch_id]=(ave_grads,max_grads)
                    optimizer.step()
                    if verbose and ((batch_id+1)%100)==0:
                        print(f"Epoch[{epoch+1}/{num_epochs}], train batch[{batch_id+1}/{len(trainloader)}], loss : {loss_value.item()}")
                if accuracy_meter is not None:
                    trainacc.append(temp_acc/len(trainloader.dataset))
                trainlosses.append(np.mean(temp_losses))

            if time_perf:
                end=time.perf_counter()
                times.append(end-start)

            #validation
            model.eval()
            if accuracy_meter is not None:
                temp_acc=accuracy_meter.zero()
            for batch_id, (batch_data, batch_truth) in enumerate(valloader):
                temp_losses=[]
                out=model(batch_data.to(device))
                loss_value=loss(out,batch_truth.to(device))
                if accuracy_meter is not None:
                    temp_acc += accuracy_meter(out,batch_truth.to(device))
                temp_losses.append(loss_value.detach().cpu().numpy())
                if verbose and ((batch_id+1)%10)==0:
                    print(f"Epoch[{epoch+1}/{num_epochs}], val batch[{batch_id+1}/{len(valloader)}], loss : {loss_value.item()}")
            if accuracy_meter is not None:
                valacc.append(temp_acc/len(valloader.dataset))
            vallosses.append(np.mean(temp_losses))

        #test
        model.eval()
        if accuracy_meter is not None:
                temp_acc=accuracy_meter.zero()
        for batch_id, (batch_data, batch_truth) in enumerate(testloader):
            temp_losses=[]
            out=model(batch_data.to(device))
            loss_value=loss(out,batch_truth.to(device))
            if accuracy_meter is not None:
                temp_acc += accuracy_meter(out,batch_truth.to(device))
            temp_losses.append(loss_value.detach().cpu().numpy())
            if verbose and ((batch_id+1)%10)==0:
                print(f"test batch[{batch_id+1}/{len(testloader)}], loss : {loss_value.item()}")
        testloss=np.mean(temp_losses)
        if accuracy_meter is not None:
            testacc=temp_acc/len(testloader.dataset)

        result_dict["trainlosses"]=trainlosses
        result_dict["vallosses"]=vallosses
        result_dict["testloss"]=testloss

        if time_perf:
            result_dict["times"]=times

        if accuracy_meter is not None:
            result_dict["trainacc"]=np.vstack(trainacc)
            result_dict["valacc"]=np.vstack(valacc)
            result_dict["testacc"]=testacc

        if track_grads:
            result_dict["grad"]=grad_dict

        return result_dict

def print_results(value_dict : Dict, hierarchy_size : Optional[Sequence]=None, ignore_first : bool=False) -> None:
    r"""Print loss and accuracy after training.
    Can print hierarchical accuracy, level 0 corresponds to leaf labels. Higher levels correspond to more general classes.
    ``ignore_first`` can be used in case no prediction on leaf labels is done.
    """
    plt.plot(value_dict["trainlosses"], label="train")
    plt.plot(value_dict["vallosses"], label="val")
    plt.legend()
    plt.show()
    print("Final loss on test set :", value_dict["testloss"])
    if "trainacc" in value_dict:
        if hierarchy_size is None:
            plt.plot(value_dict["trainacc"][:,0], label="train")
            plt.plot(value_dict["valacc"][:,0], label="val")
            plt.legend()
            plt.show()
            print("Final accuracy on test set :", value_dict["testacc"])
        else:
            for i in range(1 if ignore_first else 0, len(hierarchy_size)):
                plt.plot(value_dict["trainacc"][:,i], label="train")
                plt.plot(value_dict["valacc"][:,i], label="val")
                plt.legend()
                plt.show()
                print(f"Final accuracy (level of hierarchy : {i}) on test set :", value_dict["testacc"][i])

def plot_grad_flow(value_dict : Dict,height : Union[int,float]=5, width : Union[int,float]=7, upper_bound : Union[int,float]=0.02) -> None:
    r"""Allow an interactive plot of the gradient received by the parameters of the model.
    Epoch and batch can be selected through sliders.
    Due to interactivity of matplotlib required for this to work properly, it may encounter problems with some IDE.
    """
    grad_dict=value_dict["grad"]
    n=len(grad_dict["layers"])
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, gridspec_kw={'width_ratios': [200, 1, 1]})
    fig.set_figheight(height)
    fig.set_figwidth(width)
    ax1.set_xlabel("Layers")
    ax1.set_ylabel("Average gradient")
    max_val=max(grad_dict[0,0][1])
    max_bar=ax1.bar(np.arange(n), grad_dict[0,0][1], alpha=0.5, lw=1, color="c", label="Max gradient")
    ave_bar=ax1.bar(np.arange(n), grad_dict[0,0][0], alpha=0.5, lw=1, color="b", label="Mean gradient")
    ax1.hlines(0, 0, n+1, lw=2, color="k" )
    ax1.set_xticks(range(0,n , 1), grad_dict["layers"], rotation="vertical")
    ax1.set_xlim(left=0, right=n)
    ax1.set_ylim(bottom = -upper_bound/20, top=upper_bound)
    ax1.set_title("Gradient flow")
    ax1.grid(True)
    ax1.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['Max gradient', 'Mean gradient', 'Zero gradient'],
                ncol=3, loc="best")#, bbox_to_anchor=(1.2, 1.2), mode="expand", borderaxespad=0)


    epoch_slider = Slider(
        ax=ax2,
        label="Epoch",
        valmin=0,
        valmax=grad_dict["epochs"]-1,
        valinit=0,
        valstep=1,
        orientation="vertical"
    )

    batch_slider = Slider(
        ax=ax3,
        label=" Batch",
        valmin=0,
        valmax=grad_dict["batches"]-1,
        valinit=0,
        valstep=1,
        orientation="vertical"
    )

    def update(val):
        for i in range(n):
            max_bar[i].set_height(grad_dict[epoch_slider.val,batch_slider.val][1][i])
            ave_bar[i].set_height(grad_dict[epoch_slider.val,batch_slider.val][0][i])
        fig.canvas.draw_idle()

    epoch_slider.on_changed(update)
    batch_slider.on_changed(update)

    plt.tight_layout(w_pad=0)
    plt.show()