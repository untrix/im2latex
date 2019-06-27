# Notes about the Framework
1. Input is streamed, not loaded at once into memory
2. Use 'n' GPUs
3. Snapshots are taken every 'n' epochs or based on other dynamic conditions (e.g. best observed validation accuracy)
4. Snapshot taken when model training is interrupted
5. Metrics viewed in tensorboard
6. All hyperparameters are saved alongside model weights
7. Very flexible class for specifying hyperparameters (includes model architecture as well as training parameters)
8. ...