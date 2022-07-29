from read_data import load_data
from train import pipeline
from recommendation import get_recommendation
from read_data import load_data
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Training the model')
parser.add_argument('--model', type=str, default='GCN',
                    help='GCN/SAGE/GAT/CHEB/AGNN/GMM')
parser.add_argument('--embed_dim', type=int, default=16,
                    help='Embedding dimension')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate')

(u, v, g, num_nodes, train_g, train_pos_g,
 train_neg_g, test_pos_g, test_neg_g) = load_data()

args = parser.parse_args()

if(args.model == 'GCN'):
    h = pipeline(train_g, train_pos_g, train_neg_g, test_pos_g,
                 test_neg_g, 'GCN', hidden_size=args.embed_dim, epochs=args.epochs, lr=args.lr)
elif(args.model == 'SAGE'):
    h = pipeline(train_g, train_pos_g, train_neg_g, test_pos_g,
                 test_neg_g, 'SAGE', hidden_size=args.embed_dim, epochs=args.epochs, lr=args.lr)
elif (args.model == 'GAT'):
    h = pipeline(train_g, train_pos_g, train_neg_g, test_pos_g,
                 test_neg_g, 'GAT', hidden_size=args.embed_dim, epochs=args.epochs, lr=args.lr)
elif (args.model == 'CHEB'):
    h = pipeline(train_g, train_pos_g, train_neg_g, test_pos_g,
                 test_neg_g, 'CHEB', hidden_size=args.embed_dim, epochs=args.epochs, lr=args.lr)
elif (args.model == 'AGNN'):
    h = pipeline(train_g, train_pos_g, train_neg_g, test_pos_g,
                 test_neg_g, 'AGNN', hidden_size=args.embed_dim, epochs=args.epochs, lr=args.lr)


get_recommendation(u, v, g, num_nodes, h, user_id=4)
