# frontend/app.py
import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
import tempfile
import requests
import os
import pandas as pd # Import pandas for displaying table

# === Configuration ===
# API_BASE points to the FastAPI backend service running locally.
API_BASE = "http://localhost:8000"
# For local execution, Streamlit handles paths relative to the app.py location.
# AUDIO_BASE_URL = "http://localhost:8501/static" # This line is now effectively informational

# Set Streamlit page configuration
st.set_page_config(page_title="🎤 AI Interview System", layout="centered")
st.title("🎤 AI Interview System")

# === Session State Initialization ===
# Initialize all necessary session state variables to manage the interview flow.
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "question_index" not in st.session_state:
    st.session_state.question_index = 0
if "transcripts" not in st.session_state:
    st.session_state.transcripts = []
if "interview_complete" not in st.session_state:
    st.session_state.interview_complete = False
if "job_roles" not in st.session_state:
    st.session_state.job_roles = [] # Stores fetched job roles
if "selected_job_role_id" not in st.session_state:
    st.session_state.selected_job_role_id = None # ID of the currently selected job role
if "current_question_text" not in st.session_state:
    st.session_state.current_question_text = None # Text of the current question
if "current_audio_path" not in st.session_state:
    st.session_state.current_audio_path = None # Relative path to the current question's audio
if "job_role_name" not in st.session_state:
    st.session_state.job_role_name = None # Name of the selected job role
if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = None # To display the transcription of the last answer
if "evaluation_report" not in st.session_state:
    st.session_state.evaluation_report = None # To store the generated evaluation report

# === Function to Fetch Job Roles from Backend ===
@st.cache_data(ttl=3600) # Cache the result for 1 hour to avoid repeated API calls
def fetch_job_roles():
    """
    Fetches the list of available job roles from the FastAPI backend.
    Handles connection errors and other request exceptions.
    Returns:
        List[Dict]: A list of job role dictionaries, or an empty list if fetching fails.
    """
    try:
        res = requests.get(f"{API_BASE}/job_roles")
        res.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        return res.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend. Please ensure the backend service is running at http://localhost:8000.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching job roles: {e}")
        return []

# Fetch job roles if not already in session state
if not st.session_state.job_roles:
    st.session_state.job_roles = fetch_job_roles()
    if not st.session_state.job_roles:
        st.stop() # Stop execution if job roles cannot be fetched, as the app cannot proceed.

# === Audio Recording Function ===
def record_audio(duration=6, fs=44100):
    """
    Records audio from the microphone for a specified duration.
    Args:
        duration (int): Duration of recording in seconds.
        fs (int): Sampling frequency (samples per second).
    Returns:
        Tuple[int, numpy.ndarray]: Sampling frequency and the recorded audio data.
    """
    st.info(f"🎹 Recording for {duration} seconds...")
    # Record audio using sounddevice
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait() # Wait until recording is finished
    return fs, recording

# --- Job Role Selection UI ---
# This section is displayed when no interview session is active.
if st.session_state.session_id is None and not st.session_state.interview_complete:
    st.subheader("Select a Job Role to Start")
    # Create a selectbox with job role names
    role_names = [role["name"] for role in st.session_state.job_roles]
    # Ensure there's a default selection if roles are available
    if role_names:
        selected_role_name = st.selectbox("Choose a Job Role:", role_names, index=0)
    else:
        selected_role_name = None
        st.warning("No job roles available. Please check the backend and questions.json.")


    # Update the selected job role ID based on the selection
    if selected_role_name:
        selected_role = next((role for role in st.session_state.job_roles if role["name"] == selected_role_name), None)
        if selected_role:
            st.session_state.selected_job_role_id = selected_role["id"]

    # Button to start the interview
    if st.button("▶ Start Interview"):
        if st.session_state.selected_job_role_id:
            try:
                # Make a POST request to the backend to start the interview
                res = requests.post(f"{API_BASE}/interview/start", data={"job_role_id": st.session_state.selected_job_role_id})
                res.raise_for_status() # Check for HTTP errors
                data = res.json()

                # Update session state with interview details from the backend
                st.session_state.session_id = data.get("session_id")
                st.session_state.job_role_name = data.get("job_role_name")
                st.session_state.current_question_text = data.get("question") # Text is still retrieved for backend use
                st.session_state.current_audio_path = data.get("audio_path")
                st.session_state.question_index = 0 # Reset question index for a new interview
                st.session_state.transcripts = [] # Clear previous transcripts
                st.session_state.interview_complete = False # Mark interview as not complete
                st.session_state.last_transcription = None # Clear previous transcription display
                st.session_state.evaluation_report = None # Clear any previous report
                st.success(f"✅ Interview for {st.session_state.job_role_name} started!")
                st.rerun() # Rerun the Streamlit app to update UI
            except requests.exceptions.RequestException as e:
                st.error(f"Error starting interview: {e}")
        else:
            st.warning("Please select a job role.")

# === Main Interview Flow ===
# This section is displayed when an interview session is active and not completed.
if st.session_state.session_id and not st.session_state.interview_complete:
    st.subheader(f"🔊 Question {st.session_state.question_index + 1} for {st.session_state.job_role_name}")

    # Display the transcription of the *previous* answer, if available
    if st.session_state.last_transcription:
        st.info(f"Your last answer was: \"{st.session_state.last_transcription}\"")

    # Play the audio for the current question
    if st.session_state.current_audio_path:
        # Construct the path relative to the Streamlit app's root directory (frontend/)
        # This creates an absolute path to the audio file for st.audio
        # os.path.dirname(__file__) gets the directory of the current script (app.py)
        # Then we join it with 'static' and the rest of the audio_path from JSON
        full_audio_file_path = os.path.join(os.path.dirname(__file__), "static", st.session_state.current_audio_path)
        
        # Check if the file exists before attempting to play
        if os.path.exists(full_audio_file_path):
            st.audio(full_audio_file_path, format="audio/wav")
        else:
            st.error(f"Audio file not found: {full_audio_file_path}. Please check your folder structure and file names.")
            st.info("Ensure your audio files are in `frontend/static/job_roles/{JobRoleName}/Q#.wav`")
    else:
        st.info("No audio path specified for this question.")

    # Slider to select recording duration
    duration = st.slider("⏱ Recording Duration", 3, 15, 6)

    # Button to record and submit the answer
    if st.button("🎹 Record & Submit Answer"):
        fs, recording = record_audio(duration) # Record audio

        # Use a temporary file to save the recording
        # The 'delete=False' is crucial here to prevent immediate deletion before reading.
        # We will manually delete it in the finally block.
        tmp_file_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmp_file_path = tmpfile.name # Store the path
                scipy.io.wavfile.write(tmpfile.name, fs, recording)
                # It's important to close the file handle explicitly before sending it
                # or attempting to delete it. The 'with' statement handles this.

            # Prepare files and data for the POST request to the backend
            # Open the file again in binary read mode for sending
            with open(tmp_file_path, "rb") as audio_file_to_send:
                files = {"audio": (os.path.basename(tmp_file_path), audio_file_to_send, "audio/wav")}
                data = {"session_id": st.session_state.session_id}
                
                res = requests.post(f"{API_BASE}/interview/answer", files=files, data=data)
                res.raise_for_status() # Check for HTTP errors
                response_data = res.json()

                transcription = response_data.get("transcription", "(no transcription)")
                st.session_state.transcripts.append(transcription) # Add transcription to session state list
                st.session_state.last_transcription = transcription # Store for immediate display

                if response_data.get("completed"):
                    # If interview is complete, update state and show completion message
                    st.session_state.interview_complete = True
                    st.info("🎉 Interview Completed!")
                else:
                    # If not complete, update with details for the next question
                    st.session_state.question_index = response_data.get("question_index")
                    st.session_state.current_question_text = response_data.get("next_question")
                    st.session_state.current_audio_path = response_data.get("next_audio_path")
                st.rerun() # Rerun to display the next question or completion message

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Submission failed: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred during audio processing: {e}")
        finally:
            # Ensure the temporary file is removed regardless of success or failure
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path) # Clean up the temporary audio file

# === Interview Completion UI ===
# This section is displayed when the interview is marked as complete.
if st.session_state.interview_complete:
    st.subheader("🌟 Interview Completed")
    st.write("Thank you for completing the interview.")
    
    # Display all recorded transcriptions
    st.subheader("Your Responses:")
    for i, transcript in enumerate(st.session_state.transcripts):
        st.write(f"**Answer {i+1}:** {transcript}")

    # Button to generate report
    if st.button("📊 Generate Interview Report"):
        with st.spinner("Generating report... This may take a moment."):
            try:
                report_res = requests.get(f"{API_BASE}/interview/get_report/{st.session_state.session_id}")
                report_res.raise_for_status()
                st.session_state.evaluation_report = report_res.json()
                st.success("Report generated successfully!")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to generate report: {e}")
            st.rerun() # Rerun to display the report

    # Display the generated report if available
    if st.session_state.evaluation_report:
        report = st.session_state.evaluation_report
        if "error" in report:
            st.error(f"Error in report: {report['error']}")
        else:
            st.subheader("📝 Interview Evaluation Report")
            
            st.write("### Trait Scores (Scale: 1-Poor, 5-Excellent)")
            scores_data = [{"Trait": trait, "Score": score} for trait, score in report["scores"].items()]
            scores_df = pd.DataFrame(scores_data)
            scores_df.set_index("Trait", inplace=True)
            st.table(scores_df) # Use st.table for a simple, clear display

            st.write("### Overall Summary")
            st.markdown(report["overall_summary"]) # Use markdown for summary to allow formatting

    # Button to start a new interview, which resets the session state
    if st.button("Start New Interview"):
        # Reset all relevant session state variables to restart the flow
        keys_to_clear = ["session_id", "question_index", "transcripts",
                         "interview_complete", "selected_job_role_id",
                         "current_question_text", "current_audio_path",
                         "job_role_name", "last_transcription", "evaluation_report"] # Clear report too
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun() # Rerun to go back to job role selection
