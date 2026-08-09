# Libraries
import numpy as np
import joblib
import cherrypy


class InstagramApp:
    def __init__(self):
        # Load model
        self.model = joblib.load("../datasets/instagram_model.pkl")

        # Đọc giao diện
        with open("index.html", "r", encoding="utf-8") as file:
            self.html = file.read()

    @cherrypy.expose
    def index(self):
        return self.html

    @cherrypy.expose
    def predict(
        self,
        likes=None,
        saves=None,
        comments=None,
        shares=None,
        profile_visits=None,
        follows=None,
        captions=None
    ):
        try:
            # Convert HTML to Integer
            likes = float(likes)
            saves = float(saves)
            comments = float(comments)
            shares = float(shares)
            profile_visits = float(profile_visits)
            follows = float(follows)

            # Create features
            features = np.array([[likes, saves, comments, shares, profile_visits, follows]])

            # Predict
            prediction = self.model.predict(features)[0]

            # Show information
            result = f"""
                <div class="result">
                    <h2>Predicted Impressions</h2>
                    <p>{round(prediction):,}</p>
                </div>
            """
            return self.html.replace("RESULT", result)
        
        except Exception as e:
            error = f"""
                <div class="error">
                    Error: {str(e)}
                </div>
            """
            return self.html.replace("RESULT", error)


if __name__ == "__main__":
    # Run: python app.py
    cherrypy.config.update({
        "server.socket_host": "127.0.0.1",
        "server.socket_port": 8080
    })

    cherrypy.quickstart(
        InstagramApp()
    )
