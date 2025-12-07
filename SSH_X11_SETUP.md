# Setting Up X11 Forwarding for SSH

To see the browser window when running on a remote server via SSH, you need to enable X11 forwarding.

## Step 1: On Your Local Machine

### For Linux:
X11 is usually already available. Just make sure you have an X server running.

### For macOS:
Install XQuartz:
```bash
brew install --cask xquartz
```
Then restart your terminal and log out/in.

### For Windows:
Install one of these:
- **VcXsrv** (recommended): https://sourceforge.net/projects/vcxsrv/
- **Xming**: https://sourceforge.net/projects/xming/

After installing, start the X server before connecting via SSH.

## Step 2: Connect with X11 Forwarding

**Disconnect your current SSH session** and reconnect with one of these commands:

```bash
# Standard X11 forwarding
ssh -X user@your-server

# Trusted X11 forwarding (less secure but sometimes needed)
ssh -Y user@your-server
```

Replace `user@your-server` with your actual SSH connection details.

## Step 3: Verify X11 Forwarding is Working

Once connected, check:
```bash
echo $DISPLAY
```

You should see something like `localhost:10.0` or similar (not empty).

Test with a simple X application:
```bash
xeyes  # If installed, or
xclock  # If installed
```

If these work, X11 forwarding is set up correctly!

## Step 4: Run Your Script

Now you can run with `headless=False`:
```bash
cd /root/REAL/agisdk
python starter.py
```

The browser window should appear on your local machine!

## Troubleshooting

**If `$DISPLAY` is empty after reconnecting:**
- Make sure you used `-X` or `-Y` flag
- Check `/etc/ssh/sshd_config` on the server has:
  ```
  X11Forwarding yes
  ```
- Restart SSH service on server if needed: `sudo systemctl restart sshd`

**If you get "cannot connect to X server" errors:**
- Make sure X server is running on your local machine
- Try `-Y` instead of `-X` (trusted forwarding)
- On macOS, make sure XQuartz is running and you've logged out/in after installing

**If it's still not working:**
- Use `xvfb-run` as a fallback (browser runs but you won't see it)
- Or set `headless=True` in your script

