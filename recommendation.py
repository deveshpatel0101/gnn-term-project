import dgl
from link_prediction.dot_predictor import DotPredictor


def get_recommendation(u, v, g, num_nodes, h, user_id=0):
    user_friends = set()
    user_neg_u, user_neg_v = [], []
    for n1, n2 in zip(u, v):
        if int(n1) == user_id:
            user_friends.add(int(n2))
        if int(n2) == user_id:
            user_friends.add(int(n1))

    for i in range(num_nodes):
        if i != user_id and i not in user_friends:
            user_neg_u.append(user_id)
            user_neg_v.append(i)

    user_g = dgl.graph((user_neg_u, user_neg_v), num_nodes=g.number_of_nodes())

    pred = DotPredictor()

    scores = [(i, score) for i, score in enumerate(pred(user_g, h))]

    scores.sort(key=lambda x: -x[1])

    print(f"List of 5 suggested friends for user {user_id}:")
    for i in range(5):
        print(f'- User {scores[i][0]}, score = {scores[i][1]}')
