"""
Box Authentication Helper
Handles OAuth authentication and credential management
"""

import os
import webbrowser
from boxsdk import OAuth2


def load_credentials(filename='box_credentials.txt'):
    """
    Load credentials from file

    Args:
        filename: Path to credentials file

    Returns:
        Dictionary with credentials or None if file doesn't exist
    """
    if not os.path.exists(filename):
        return None

    credentials = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '=' in line:
                key, value = line.split('=', 1)
                credentials[key.strip()] = value.strip()

    return credentials


def save_credentials(credentials, filename='box_credentials.txt'):
    """
    Save credentials to file

    Args:
        credentials: Dictionary with credentials
        filename: Path to save credentials
    """
    with open(filename, 'w') as f:
        for key, value in credentials.items():
            f.write(f"{key}={value}\n")
    print(f"\nCredentials saved to: {filename}")


def get_access_token(client_id, client_secret, redirect_uri='http://localhost:3000'):
    """
    Guide user through OAuth flow to get access token

    Args:
        client_id: Box application client ID
        client_secret: Box application client secret
        redirect_uri: OAuth redirect URI

    Returns:
        Tuple of (access_token, refresh_token) or (None, None) if failed
    """
    print("=" * 70)
    print("BOX OAUTH AUTHENTICATION")
    print("=" * 70)

    try:
        oauth = OAuth2(
            client_id=client_id,
            client_secret=client_secret
        )

        # Get authorization URL
        auth_url, csrf_token = oauth.get_authorization_url(redirect_uri)

        print("\nSTEP 1: Authorize the application")
        print("-" * 70)
        print("\nOpening browser for authorization...")
        print("If it doesn't open, copy this URL:\n")
        print(auth_url)
        print()

        # Open browser
        try:
            webbrowser.open(auth_url)
            print("Browser opened successfully!")
        except:
            print("Could not open browser automatically.")
            print("Please copy the URL above and paste it in your browser.")

        print("\nSTEP 2: Get the authorization code")
        print("-" * 70)
        print("\nAfter authorizing:")
        print("1. You'll be redirected to a page that may not load")
        print("2. Look at the URL in your browser")
        print("3. It will look like: http://localhost:3000/?code=XXXXX")
        print("4. Copy everything after 'code=' (the authorization code)")
        print()

        auth_code = input("Enter the authorization code: ").strip()

        print("\nAuthenticating...")
        access_token, refresh_token = oauth.authenticate(auth_code)

        print("\n" + "=" * 70)
        print("SUCCESS! Authentication complete")
        print("=" * 70)

        return access_token, refresh_token

    except Exception as e:
        print(f"\nError during authentication: {str(e)}")
        print("\nCommon issues:")
        print("- Make sure the redirect URI in your Box app matches:", redirect_uri)
        print("- Make sure you copied the full authorization code")
        print("- Check that your Client ID and Secret are correct")
        return None, None


def authenticate_or_refresh(credentials_file='box_credentials.txt'):
    """
    Load existing credentials or guide user through authentication

    Args:
        credentials_file: Path to credentials file

    Returns:
        Dictionary with CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN
        or None if authentication fails
    """
    credentials = load_credentials(credentials_file)

    if credentials is None:
        print("\nNo credentials file found!")
        print("\nTo set up authentication:")
        print("1. Get your CLIENT_ID and CLIENT_SECRET from Box Developer Console")
        print("2. Create a file named 'box_credentials.txt' with:")
        print("   CLIENT_ID=your_client_id")
        print("   CLIENT_SECRET=your_client_secret")
        print("3. Run this script again")
        return None

    # Check for required credentials
    required = ['CLIENT_ID', 'CLIENT_SECRET']
    missing = [key for key in required if key not in credentials]

    if missing:
        print(f"\nMissing credentials: {', '.join(missing)}")
        print("\nPlease add them to box_credentials.txt")
        return None

    # Check if we need to get tokens
    if 'ACCESS_TOKEN' not in credentials or 'REFRESH_TOKEN' not in credentials:
        print("\nNo access tokens found. Starting OAuth flow...")
        access_token, refresh_token = get_access_token(
            credentials['CLIENT_ID'],
            credentials['CLIENT_SECRET']
        )

        if access_token and refresh_token:
            credentials['ACCESS_TOKEN'] = access_token
            credentials['REFRESH_TOKEN'] = refresh_token
            save_credentials(credentials, credentials_file)
        else:
            print("\nAuthentication failed!")
            return None

    return credentials
