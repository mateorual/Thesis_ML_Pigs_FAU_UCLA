# Box Basic Data Analysis - Complete Setup Guide

This guide will walk you through setting up and running the Box Basic Data Analysis tool from scratch.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Getting Box API Credentials](#getting-box-api-credentials)
3. [Cloning the Repository](#cloning-the-repository)
4. [Setting Up the Python Environment](#setting-up-the-python-environment)
5. [Configuring Box Credentials](#configuring-box-credentials)
6. [Running the Analysis](#running-the-analysis)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.7 or higher**
  - Check your version: `python --version` or `python3 --version`
  - Download from: https://www.python.org/downloads/

- **Git** (for cloning the repository)
  - Check if installed: `git --version`
  - Download from: https://git-scm.com/downloads

- **Box Account** with access to the data folders
  - You'll need developer access to create an application

---

## Getting Box API Credentials

### Step 1: Create a Box Application

1. Go to the **Box Developer Console**: https://app.box.com/developers/console

2. Click **"Create New App"**

3. Select **"Custom App"** as the app type

4. Choose **"Standard OAuth 2.0 (User Authentication)"** as the authentication method

5. Give your app a name (e.g., "Box Audio Analysis Tool")

### Step 2: Configure Your Application

1. After creating the app, you'll be taken to the **Configuration** page

2. Scroll down to **"OAuth 2.0 Redirect URI"** and add:
   ```
   http://localhost:3000
   ```

3. Scroll down to **"Application Scopes"** and ensure these are selected:
   - ✅ Read all files and folders stored in Box
   - ✅ Write all files and folders stored in Box (optional, but recommended)

4. Click **"Save Changes"** at the top of the page

### Step 3: Get Your Credentials

1. At the top of the Configuration page, you'll see:
   - **Client ID**: A long alphanumeric string
   - **Client Secret**: Click "Show" to reveal it

2. **IMPORTANT**: Copy both of these values. You'll need them in the next steps.

3. Keep these credentials **SECRET** - never share them or commit them to version control!

---

## Cloning the Repository

1. Open a terminal/command prompt

2. Navigate to where you want to store the project:
   ```bash
   cd /path/to/your/projects
   ```

3. Clone the repository:
   ```bash
   git clone https://github.com/mateorual/Thesis_ML_Pigs_FAU_UCLA.git
   ```

4. Navigate to the project directory:
   ```bash
   cd Thesis_ML_Pigs_FAU_UCLA/UCLA_Shared/Box_Basic_Data_Analysis
   ```

---

## Setting Up the Python Environment

### Option 1: Using venv (Recommended)

1. Create a virtual environment:
   ```bash
   # On Windows
   python -m venv venv

   # On macOS/Linux
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using conda

1. Create a conda environment:
   ```bash
   conda create -n box_analysis python=3.9
   ```

2. Activate the environment:
   ```bash
   conda activate box_analysis
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuring Box Credentials

### Step 1: Create the Credentials File

1. In the `Box_Basic_Data_Analysis` directory, create a file named `box_credentials.txt`

2. You can use the provided template:
   ```bash
   # On Windows
   copy box_credentials_example.txt box_credentials.txt

   # On macOS/Linux
   cp box_credentials_example.txt box_credentials.txt
   ```

   Or create it manually with a text editor.

### Step 2: Add Your Credentials

1. Open `box_credentials.txt` in a text editor

2. Replace the placeholder values with your actual credentials:
   ```
   CLIENT_ID=your_actual_client_id_here
   CLIENT_SECRET=your_actual_client_secret_here
   ```

3. **Do NOT add ACCESS_TOKEN or REFRESH_TOKEN yet** - these will be generated automatically on first run

4. Save the file

### Important Notes:
- The file must be named exactly `box_credentials.txt`
- Do NOT add quotes around the values
- Do NOT add spaces around the `=` sign
- Each credential should be on its own line

---

## Running the Analysis

### First Run - OAuth Authentication

1. Make sure your virtual environment is activated (you should see `(venv)` or `(box_analysis)` in your terminal)

2. Run the main script:
   ```bash
   python main.py
   ```

3. The script will detect that you don't have access tokens yet and will:
   - Open your default web browser
   - Take you to Box's authorization page
   - Ask you to log in and authorize the application

4. After authorizing:
   - You'll be redirected to a page that may show "This site can't be reached" or similar
   - **This is normal!** Look at the URL in your browser's address bar
   - Copy the **entire URL** (it will start with `http://localhost:3000/?code=...`)

5. Return to your terminal and paste the URL when prompted

6. The script will:
   - Exchange the authorization code for access tokens
   - Save the tokens to `box_credentials.txt`
   - Begin the analysis automatically

### Subsequent Runs

For all future runs, simply execute:
```bash
python main.py
```

The script will automatically:
- Use your saved tokens
- Refresh them if they've expired
- Run the analysis without requiring re-authentication

---

## Configuration Options

Before running the analysis, you may want to customize the settings in `config.py`:

### Choose Analysis Mode

**Folder Structure:**
```python
FOLDER_STRUCTURE = "All Pig Snips"  # or "Individual Pairs DONE"
```

**Analysis Type:**
```python
ANALYSIS_TYPE = "Metadata Analysis"  # or "F0 Analysis" or "Metadata and F0 Analysis"
```

### Common Settings:
- `BOX_ROOT_FOLDER_ID`: Root folder ID in Box
- `ALL_PIG_SNIPS_FOLDER_ID`: ID for "All Pig Snips" folder
- `F0_MIN_FREQ` / `F0_MAX_FREQ`: Frequency range for F0 analysis

---

## Output

After running the analysis, you'll find the results in the `output/` directory:

### Excel Files:
- `box_audio_metadata_analysis.xlsx` - Metadata statistics and summaries
- `box_audio_f0_analysis.xlsx` - Fundamental frequency analysis results

### Visualizations (PNG files):
- `1_audio_count_per_week_by_subject.png`
- `2_total_audio_per_subject.png`
- `3_duration_distribution_by_subject.png`
- `4_total_duration_per_subject.png`
- `5_heatmap_subject_vs_week.png`

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'boxsdk'" (or similar)

**Solution:** Your virtual environment isn't activated or packages aren't installed
```bash
# Activate your environment first, then:
pip install -r requirements.txt
```

### "Invalid credentials" or "Authentication failed"

**Solutions:**
1. Verify your `CLIENT_ID` and `CLIENT_SECRET` are correct
2. Check that there are no extra spaces or quotes in `box_credentials.txt`
3. Ensure your redirect URI is exactly `http://localhost:3000` in the Box app settings
4. Delete the `ACCESS_TOKEN` and `REFRESH_TOKEN` lines from `box_credentials.txt` and re-authenticate

### "Refresh token has expired"

**Solution:** Delete the `ACCESS_TOKEN` and `REFRESH_TOKEN` lines from `box_credentials.txt` and run the script again to re-authenticate

### "No files found" during analysis

**Solutions:**
1. Verify you have access to the Box folders
2. Check that the folder IDs in `config.py` are correct
3. Ensure the folder structure matches what the script expects
4. Review the `FOLDER_STRUCTURE` setting in `config.py`

### Browser doesn't open during authentication

**Solution:** Copy the authorization URL from the terminal and paste it manually into your browser

### "Permission denied" when creating output files

**Solution:** Ensure you have write permissions in the project directory and that no Excel files are currently open

---

## Security Best Practices

1. **Never commit `box_credentials.txt` to version control**
   - It's already in `.gitignore`, but double-check before pushing

2. **Keep your credentials secure**
   - Don't share them via email or chat
   - Don't include them in screenshots

3. **Regenerate credentials if compromised**
   - Go to Box Developer Console
   - Reset your Client Secret
   - Update `box_credentials.txt`

4. **Use separate Box apps for production and testing**
   - Create different apps for different environments
   - This limits access scope if credentials are compromised

---

## Getting Help

If you encounter issues not covered in this guide:

1. Check the main [README.md](README.md) for additional documentation
2. Review error messages carefully - they often indicate the problem
3. Ensure all prerequisites are properly installed
4. Verify your Box account has access to the required folders

---

## Quick Reference - Commands Summary

```bash
# Clone repository
git clone https://github.com/mateorual/Thesis_ML_Pigs_FAU_UCLA.git
cd Thesis_ML_Pigs_FAU_UCLA/UCLA_Shared/Box_Basic_Data_Analysis

# Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Create credentials file
# Edit box_credentials.txt with your CLIENT_ID and CLIENT_SECRET

# Run analysis
python main.py
```

---

**You're all set!** The analysis should now run successfully and generate comprehensive reports about your audio data.
