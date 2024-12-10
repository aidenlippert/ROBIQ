import pyttsx3

class AudioFeedback:
    def __init__(self):
        # Initialize the TTS engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 1)  # Volume range: 0.0 to 1.0

    def speak(self, text):
        """
        Convert text to speech.
        Parameters:
        - text: The string of text to be converted into audio.
        """
        self.engine.say(text)
        self.engine.runAndWait()

    def give_feedback(self, feedback_text):
        """
        Provide auditory feedback to the user.
        Parameters:
        - feedback_text: Textual feedback to be converted into speech.
        """
        print("Feedback: ", feedback_text)  # Optionally print for debugging
        self.speak(feedback_text)


# Test Feedback Module
if __name__ == "__main__":
    from src.feedback.feedback_generator import FeedbackGenerator
    from src.feedback.audio_feedback import AudioFeedback

    def test_feedback():
        fg = FeedbackGenerator()
        af = AudioFeedback()

        # Example joint angles
        joint_angles = {'elbow': 150}

        # Generate text feedback
        feedback_text = fg.analyze_joint_angles(joint_angles)
        print(feedback_text)

        # Provide audio feedback
        af.give_feedback(feedback_text)

    test_feedback()