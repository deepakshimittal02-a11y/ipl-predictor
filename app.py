import streamlit as st
import pickle
import pandas as pd

# Load model and encoders
model         = pickle.load(open('model/ipl_model.pkl', 'rb'))
team_encoder  = pickle.load(open('model/team_encoder.pkl', 'rb'))
venue_encoder = pickle.load(open('model/venue_encoder.pkl', 'rb'))
toss_encoder  = pickle.load(open('model/toss_encoder.pkl', 'rb'))

# Load processed data — venue win rate calculate karne ke liye
df = pd.read_csv('data/processed_matches.csv')

# Teams and venues — original names ke liye
teams  = sorted(list(team_encoder.classes_))
venues = sorted(list(venue_encoder.classes_))

# ── Helper functions ──

def get_venue_win_rate(team1_enc, venue_enc):
    filtered = df[
        (df['team1'] == team1_enc) & 
        (df['venue'] == venue_enc)
    ]
    if len(filtered) == 0:
        return 0.5  # No history — neutral
    return filtered['venue_win_rate'].mean()

def get_recent_form(team_enc, is_team1=True):
    if is_team1:
        filtered = df[df['team1'] == team_enc]
        if len(filtered) == 0:
            return 0.5
        return filtered['team1_recent_form'].iloc[-1]
    else:
        filtered = df[df['team2'] == team_enc]
        if len(filtered) == 0:
            return 0.5
        return filtered['team2_recent_form'].iloc[-1]

# ── App UI ──
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏")
st.title("🏏 IPL Match Win Predictor")
st.caption("Predict the winner of an IPL match based on historical data (2008–2026)")
st.divider()

col1, col2 = st.columns(2)

with col1:
    team1        = st.selectbox("Select Team 1", teams)
    toss_winner  = st.selectbox("Toss Winner", teams)

with col2:
    team2         = st.selectbox("Select Team 2", teams)
    toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

venue = st.selectbox("Select Venue", venues)
st.divider()

# ── Prediction ──
if st.button("Predict Winner 🏆", use_container_width=True):

    if team1 == team2:
        st.error("Please select two different teams.")

    else:
        # Encode inputs
        t1 = team_encoder.transform([team1])[0]
        t2 = team_encoder.transform([team2])[0]
        tw = team_encoder.transform([toss_winner])[0]
        td = toss_encoder.transform([toss_decision])[0]
        v  = venue_encoder.transform([venue])[0]

        # Calculate engineered features
        vwr  = get_venue_win_rate(t1, v)
        t1rf = get_recent_form(t1, is_team1=True)
        t2rf = get_recent_form(t2, is_team1=False)

        # Input dataframe
        input_data = pd.DataFrame({
            'team1'            : [t1],
            'team2'            : [t2],
            'toss_winner'      : [tw],
            'toss_decision'    : [td],
            'venue'            : [v],
            'venue_win_rate'   : [vwr],
            'team1_recent_form': [t1rf],
            'team2_recent_form': [t2rf]
        })

        # Predict
        prediction = model.predict(input_data)[0]
        proba      = model.predict_proba(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(f"🏆 {team1} is predicted to WIN!")
        else:
            st.success(f"🏆 {team2} is predicted to WIN!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label=f"{team1}",
                value=f"{round(proba[1]*100, 1)}%",
                delta="Win Probability"
            )
        with col2:
            st.metric(
                label=f"{team2}",
                value=f"{round(proba[0]*100, 1)}%",
                delta="Win Probability"
            )

        # Show engineered features
        with st.expander("See match insights"):
            st.write(f"**{team1} win rate at {venue}:** {round(vwr*100, 1)}%")
            st.write(f"**{team1} recent form:** {round(t1rf*100, 1)}%")
            st.write(f"**{team2} recent form:** {round(t2rf*100, 1)}%")

        st.divider()
        st.caption("⚠️ This prediction is based on historical match data. Actual results may vary.")