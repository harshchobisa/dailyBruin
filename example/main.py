import argparse
import constants
from train import train

parser = argparse.ArgumentParser()

# add command line arguments for various settings
parser.add_argument('--train_data_path',
    type=str, required=True, help='Path to training data')
parser.add_argument('--val_data_path',
    type=str, required=True, help='Path to validation data')
parser.add_argument('--epochs',
    type=int, default=constants.EPOCHS, help='Number of epochs to train')
parser.add_argument('--batch_size',
    type=int, default=constants.BATCH_SIZE, help='Number of samples per batch')
parser.add_argument('--summary_path',
    type=str, default=constants.SUMMARY_PATH, help='Path to store tensorboard logs')
parser.add_argument('--n_summary',
    type=int, default=constants.N_SUMMARY, help='Number steps between each training summary log to tensorboard')
parser.add_argument('--n_eval',
    type=int, default=constants.N_EVAL, help='Number steps between each evaluation")
parser.add_argument('--hyperparameter1',
    type=str, default=constants.HYPERPARAMETER1, help='Netowrk hyperparameter 1')
parser.add_argument('--hyperparameter2',
    type=str, default=constants.HYPERPARAMETER2, help='Network hyperparameter 2')
parser.add_argument('--device',
    type=str, default=constants.SUMMARY_PATH, help='Device to run training on')


args = parser.parse_args()


if __name__ == "__main__":
    # validate arguments are valid
    assert(args.batch_size > 0)
    assert(device in ['cpu', 'gpu'])

    train(train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        batch_size=args.batch_size,
        summary_path=args.summary_path,
        n_summary=args.n,
        n_eval=args.n_eval,
        hyperparameter1=args.hyperparameter1,
        hyperparameter2=args.hyperparameter2,
        device=args.device)


    