Image Classification CI Pipeline ProjectThis repository provides an automated Continuous Integration (CI) pipeline using GitHub Actions for an Image Classification application.Project Structure├── .github/
│   └── workflows/
│       └── ci.yaml         # GitHub Actions CI workflow file
├── model.py                # Image classification model & inference service
├── test.py                 # Automated unit tests
├── requirements.txt        # Project dependencies
└── README.md               # Documentation & setup instructions
Step-by-Step Setup and Execution Instructions1. Initialize Git and Commit the FilesOpen your terminal in the project directory:git init
git add .
git commit -m "feat: setup image classification model, test suite, and CI pipeline"
2. Connect to GitHub and PushCreate a repository on GitHub (e.g., image-classification-ci), then run:# Rename branch to main if not already
git branch -M main

# Link your remote repository (replace with your repo URL)
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# Push to trigger GitHub Actions
git push -u origin main
Capturing the Required ScreenshotsOnce pushed, follow these steps to capture the required screenshots for submission:Actions Overview Screenshot:Go to your GitHub repository in your web browser.Click on the "Actions" tab at the top.You will see the workflow run named "Image Classification CI Pipeline" with a green checkmark ($\checkmark$) indicating success.Take a screenshot of this page showing your repository name, commit message, and the green checkmark.Job Execution Details Screenshot:Click directly on the successful workflow run.Click on the "Run Automated Tests" job on the left sidebar.Expand the "Run unit and integration tests" step to display the passing test logs (test_model_architecture_output_shape ... ok, test_predict_structure ... ok, OK).Take a screenshot displaying the expanded steps with green status icons.Creating the Final Submission ZIP FileInclude the following items inside your final ZIP archive:submission.zip
├── .github/
│   └── workflows/
│       └── ci.yaml
├── model.py
├── test.py
├── requirements.txt
├── README.md
└── screenshots/
    ├── github_actions_overview.png
    └── github_actions_job_details.png
To create the ZIP on Linux/macOS:zip -r submission.zip . -x "*.git*"
Or on Windows: Select all files $\rightarrow$ Right Click $\rightarrow$ Send to $\rightarrow$ Compressed (zipped) folder.