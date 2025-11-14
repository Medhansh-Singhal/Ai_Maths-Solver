# AI Maths Solver

This project allows users to solve mathematical and physics equations interactively by drawing them directly on the screen using a camera. Through computer vision, hand tracking, and integration with Google's Gemini API, users can visually write out problems, which are then analyzed and solved by a generative AI model.

## Features

- **Real-time Drawing Recognition:** Use your webcam and hand gestures to draw equations or problems. The system recognizes finger positions and lets you switch between brush color, eraser, and clear/reset options using gesture controls.
- **Interactive Canvas:** Write or sketch equations, math, or physics problems as you would on a whiteboard.
- **AI-Powered Solutions:** Captured images of your problem are sent to Gemini AI via API, which analyzes the image and returns solutions or related explanations directly in the app.
- **Intuitive Streamlit UI:** User interface powered by Streamlit for easy access and visibility of both your drawing canvas and AI-generated results.
- **Image Storage:** Automatically saves your drawn problems and results in organized folders.

## Applications

AI Maths Solver can be utilized in various domains, such as:
- **Education:** Teachers and students can use it to solve equations interactively, explain mathematical or physics concepts visually, or check handwritten homework and practice problems.
- **Tutoring and Online Learning:** Enables remote and interactive problem-solving, letting tutors and learners draw challenging problems and instantly receive AI-generated solutions and explanations.
- **Research & Brainstorming:** Researchers, engineers, and scientists can quickly sketch equations or systems and have them analyzed on the fly.
- **Accessibility:** Provides an alternative for those who struggle with typing, enabling equation entry via drawing and gestures.
- **Exams and Practice:** Facilitates instant checking of practice problems drawn out during study sessions, speeding up feedback and learning.
- **Creative Problem-Solving:** Useful for brainstorming and visually representing problems in study groups, hackathons, or workshops.

## How It Works

1. Launch the application, which opens up a camera feed and interactive canvas.
2. Use hand gestures to draw on the canvas. Select between drawing, colors, and erasing using intuitive finger positions.
3. Once your equation or problem is complete, the system captures the image and sends it to Gemini AI via the [google-generativeai](https://pypi.org/project/google-generativeai/) API.
4. The returned solution or explanation is displayed in the sidebar ("Results") of the application.

## Requirements

See [`requirements.txt`](https://github.com/Medhansh-Singhal/Ai_Maths-Solver/blob/main/requirements.txt) for full dependencies:

- pillow
- streamlit
- mediapipe
- opencv-python
- numpy
- google-generativeai
- dotenv

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Medhansh-Singhal/Ai_Maths-Solver.git
   cd Ai_Maths-Solver
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Gemini API key**  
   Create a `.env` file and add your Gemini (Google Generative AI) API key:
   ```
   API_KEY=your-google-api-key
   ```

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

5. **Draw your problem on the canvas, hit "Run", and view the AI-powered solution in the sidebar.**

## Usage Notes

- This project uses your computer’s webcam to track hand gestures and facilitate a natural writing experience.
- All equations and sketches are analyzed using the Gemini AI model.  
- Works best in well-lit environments for accurate hand tracking.

## Project Structure

- `main.py` : Streamlit application logic, computer vision, and AI integration.
- `requirements.txt` : List of Python dependencies.
- `/Storage` : Saved images and results (auto-generated during usage).

## Contributing

Pull requests and suggestions are welcome! Create an issue or contact the maintainer for bug reports or feature requests.


---

> _“Solve maths and physics problems with the magic of AI and computer vision — just draw them!”_
