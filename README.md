# Minimal Discord Bot for AWS EC2

This is a tiny Discord bot built with `discord.py`. It supports classic prefix commands (`!ping`, `!say`) and a slash command (`/hello`).

## 1) Discord setup (one-time)
1. Go to the Discord Developer Portal → Applications → *New Application*.
2. Add a **Bot** under *Bot* tab → *Add Bot*.
3. **Copy the token** (you'll set it as `DISCORD_TOKEN` on EC2).
4. Turn on **MESSAGE CONTENT INTENT** under *Privileged Gateway Intents*.
5. Invite the bot: under *OAuth2 → URL Generator*, pick scopes **bot** and **applications.commands**; give it a basic permission like **Send Messages**. Open the generated URL and add it to your server.

## 2) Launch an EC2 instance
- AMI: Ubuntu 22.04 LTS (or similar)
- Instance type: `t2.micro` is fine for tests
- Security group:
  - Inbound: SSH (22) from your IP. No public inbound port is required for this bot.
  - Outbound: allow all (needed to reach Discord).

## 3) Prepare the server
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv git
sudo useradd -r -m -U -s /bin/bash discordbot || true
sudo mkdir -p /opt/discord-bot
sudo chown -R discordbot:discordbot /opt/discord-bot
```

### Copy project files
Upload these files to `/opt/discord-bot` (e.g., `scp` the zip, then unzip).

```bash
cd /opt/discord-bot
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4) (Option A) Run in the foreground for a quick test
```bash
export DISCORD_TOKEN="YOUR_BOT_TOKEN_HERE"
# Optional: set your server ID to speed up slash command sync during dev
# export GUILD_ID="123456789012345678"
python bot.py
```

## 5) (Option B) Run as a systemd service (recommended)
Create `/etc/systemd/system/discord-bot.service`:
```ini
[Unit]
Description=Discord Bot
After=network.target

[Service]
Type=simple
User=discordbot
WorkingDirectory=/opt/discord-bot
Environment=DISCORD_TOKEN=YOUR_BOT_TOKEN_HERE
# Environment=GUILD_ID=123456789012345678   # optional during development
ExecStart=/opt/discord-bot/.venv/bin/python /opt/discord-bot/bot.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Then enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable discord-bot
sudo systemctl start discord-bot
sudo journalctl -u discord-bot -f
```

## Notes
- Keep your token secret. Consider storing it with `EnvironmentFile=/etc/discord-bot.env` instead of inline.
- No inbound ports are required; the bot connects out to Discord over HTTPS/WebSockets.
- If you only see prefix commands but not `/hello`:
  - Ensure the bot has the `applications.commands` scope.
  - Global slash commands can take up to an hour to appear the first time. Set `GUILD_ID` to sync to a single server instantly while developing.
