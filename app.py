import streamlit as st
import pandas as pd
import pickle
import joblib

# ---------------------------------
# Load files
# ---------------------------------

rf_recommender = joblib.load("model.pkl")

#similar_matrix = pickle.load(
#    open("similar_matrix.pkl", "rb")
#)

#user_item = pickle.load(
#    open("user_item.pkl", "rb")
#)

df_final = pickle.load(open("df_final.pkl", "rb"))

user_avg = pickle.load(
    open("user_avg.pkl", "rb")
)

prod_avg = pickle.load(
    open("prod_avg.pkl", "rb")
)

user_count = pickle.load(
    open("user_count.pkl", "rb")
)

prod_count = pickle.load(
    open("prod_count.pkl", "rb")
)

features = pickle.load(
    open("features.pkl", "rb")
)

global_mean = pickle.load(
    open("global_mean.pkl", "rb")
)

# --------------------------------------------
# Computation of Similar matrix and User item
# --------------------------------------------

user_item = df_final.pivot_table(index = 'user_id', columns = 'prod_id', values = 'rating')

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(user_item)

similar_matrix = pd.DataFrame(similarity, index = user_item.index, columns = user_item.index)


# ---------------------------------
# Recommendation Function
# ---------------------------------

def hybrid_recommendation(userid, top=10):

    similar_users = similar_matrix[userid].sort_values(
        ascending=False
    )

    similar_users = similar_users.drop(userid)

    top_sim_users = similar_users.head(10)

    already_rated = user_item.columns[
        user_item.loc[userid] > 0
    ].tolist()

    candidate_products = set()

    for user in top_sim_users.index:

        sim_user_products = user_item.columns[
            user_item.loc[user] > 0
        ]

        for prod in sim_user_products:

            if prod not in already_rated:
                candidate_products.add(prod)

    prediction_rows = []

    user_avg_val = user_avg.get(userid, global_mean)
    user_count_val = user_count.get(userid, 0)

    for prod in candidate_products:

        prod_avg_val = prod_avg.get(prod, global_mean)
        prod_count_val = prod_count.get(prod, 0)

        user_bias = user_avg_val - global_mean
        prod_bias = prod_avg_val - global_mean

        interaction = user_bias * prod_bias

        prediction_rows.append({

            'product': prod,

            'user_avg': user_avg_val,
            'prod_avg': prod_avg_val,

            'user_bias': user_bias,
            'prod_bias': prod_bias,

            'interaction': interaction,

            'user_count': user_count_val,
            'prod_count': prod_count_val

        })

    pred_df = pd.DataFrame(prediction_rows)

    pred_df['score'] = rf_recommender.predict_proba(
        pred_df[features]
    )[:, 1]

    final_recommendations = pred_df.sort_values(
        by='score',
        ascending=False
    )

    return final_recommendations[
        ['product', 'score']
    ].head(top)

# ---------------------------------
# Streamlit UI
# ---------------------------------

st.title("Hybrid Recommendation System")

st.write(
    "Recommendation System using Collaborative Filtering + Random Forest"
)

user_list = list(user_item.index)

selected_user = st.selectbox(
    "Select User",
    user_list
)

top_n = st.slider(
    "Number of Recommendations",
    1,
    20,
    10
)

if st.button("Get Recommendations"):

    recommendations = hybrid_recommendation(
        selected_user,
        top_n
    )

    st.subheader("Recommended Products")

    st.dataframe(recommendations)
