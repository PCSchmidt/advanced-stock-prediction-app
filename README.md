# Advanced Stock Price Prediction App

[... Keep the existing content up to the "Deploying to Heroku" section ...]

## Deploying to Heroku

To deploy this application to Heroku, follow these steps:

1. Make sure you have a Heroku account and the Heroku CLI installed.

2. Log in to Heroku CLI:
   ```
   heroku login
   ```

3. Create a new Heroku app:
   ```
   heroku create your-app-name
   ```

4. Set the Python buildpack:
   ```
   heroku buildpacks:set heroku/python
   ```

5. Choose one of the following deployment methods:

   Option A: Deploy from the main branch (recommended for production)
   - If you're currently on the dev branch, first merge your changes to the main branch:
     ```
     git checkout main
     git merge dev
     git push origin main
     ```
   - Then deploy to Heroku:
     ```
     git push heroku main
     ```

   Option B: Deploy directly from the dev branch
   - If you want to deploy your current dev branch without merging to main:
     ```
     git push heroku dev:main
     ```

6. Scale the web dyno:
   ```
   heroku ps:scale web=1
   ```

7. Open the deployed application in your browser:
   ```
   heroku open
   ```

Note: Make sure all the necessary files (Procfile, runtime.txt, requirements.txt) are present and committed in your current branch before deploying to Heroku.

Feel free to contribute to this project by submitting pull requests or opening issues for any bugs or feature requests.
