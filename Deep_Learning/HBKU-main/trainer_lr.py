# Extra
def get_lr_trainer(model, trainer, loss):
    if loss == 'asymm':
        new_lr = 1e-8
        model.lr = new_lr
        print(f'recommended LR for {loss}  is: {new_lr}')

    else:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model)
        # Results can be found in
        lr_finder.results
        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr_finder_" + loss)
        # fig.show()
        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        # update hparams of the model
        model.lr = new_lr
        print(f'recommended LR for {loss}  is: {new_lr}')

    # 8.317637711026709e-07
    return model


def find_bs(model, trainer):
    trainer.tune(model)
