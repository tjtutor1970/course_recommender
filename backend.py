import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import numpy as np
import recommendernet as rn

models = ("Course Similarity",
          "Neural Network")


def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")

def load_nn_user_idx2id_dict():
    with open('nn_user_idx2id_dict.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def load_nn_course_idx2id_dict():
    with open('nn_course_idx2id_dict.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def load_nn_ratings():
    return pd.read_csv("nn_ratings.csv")


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


# Model training
def train(model_name, params):
    # TODO: Add model training code here
    pass


# Prediction
def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    top_courses = 10
    if 'top_courses' in params:
        top_courses = params['top_courses']
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)
        # TODO: Add prediction model code here
        # Neural Network model
        if model_name == models[1]:
            # load model and get latent features
            nn_model = tf.keras.models.load_model('nn_model.keras', custom_objects={"RecommenderNet": rn.RecommenderNet})
            user_latent_features = nn_model.get_layer('user_embedding_layer').get_weights()[0]
            item_latent_features = nn_model.get_layer('item_embedding_layer').get_weights()[0]

            ratings_df = load_ratings() # this is what the user entered on UI
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            user_idx2id_dict = load_nn_user_idx2id_dict()

            # check if the user_id is in user_idx2id_dict
            if len([u_key for u_key, u_id in user_idx2id_dict.items() if u_id == user_id]) > 0:
                print(f"user id {user_id} is one of the users used to train nn model.")
            else:
                print(f"user id {user_id} is NOT one of the users used to train nn model.")
                user_courses = set(user_ratings['item'].unique())
                # print(f"user courses: {user_courses}")
                nn_ratings = load_nn_ratings() # this is what was used to train nn model
                user_course_ratings = nn_ratings[nn_ratings['item'].isin(user_courses)] # identify other users who took the same courses            
                similar_users = set(user_course_ratings['user'].unique())
                # print(f"user course ratings has {len(similar_users)} rows.")           
                user_idx2id_dict = load_nn_user_idx2id_dict()
                user_indexes = [u_key for u_key, u_id in user_idx2id_dict.items() if u_id in similar_users]
                similar_courses = nn_ratings[nn_ratings['user'].isin(similar_users)]
                # print(f"similar courses has {len(similar_courses)} rows.")           
                course_idx2id_dict = load_nn_course_idx2id_dict()
                similar_course_ids = set(similar_courses['item'].unique())
                course_indexes = [course_key for course_key, course_id in course_idx2id_dict.items() if course_id in similar_course_ids]
                user_course_indexes = [course_key for course_key, course_id in course_idx2id_dict.items() if course_id in user_courses]
                
                candidate_courses = []
                for u_i in user_indexes:
                    A = user_latent_features[u_i,:].reshape((1,-1))
                    for c_i in course_indexes:
                        if c_i in user_course_indexes:
                            continue
                        B = item_latent_features[c_i,:].T.reshape(-1,1)
                        C = np.dot(A, B)
                        candidate_courses.append((u_i,c_i,C[0,0]))
                        
                cc_df = pd.DataFrame(candidate_courses, columns = ['user_idx', 'item_idx', 'rating'])
                cc_grp = cc_df.groupby('item_idx')['rating'].max().reset_index()
                top_X = cc_grp.sort_values(by='rating',ascending=False)[:top_courses]
                course_ids = [course_idx2id_dict.get(course_key) for course_key in top_X.item_idx]
                users = [user_id] * len(course_ids)
                courses = course_ids
                scores = top_X.rating.astype(float)

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df
