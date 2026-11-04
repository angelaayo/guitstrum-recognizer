# Deploying the recognizer to a free Oracle Cloud VM

Goal: run this FastAPI WebSocket service for free on an Oracle Cloud "Always Free"
Arm VM, reachable from the browser over `wss://` (required, because the GuitStrum
frontend is served over HTTPS). Caddy obtains a free Let's Encrypt TLS certificate
automatically.

Everything in steps 1–4 is done in the Oracle Cloud web console and your browser.
Steps 5–8 are commands you run on the VM over SSH.

---

## 1. Create an Oracle Cloud account
- Sign up at https://signup.cloud.oracle.com. A card is required for identity
  verification, but **Always Free** resources never incur charges.
- Pick a home region near you (you can only create Always Free resources in your
  home region, so choose carefully).

## 2. Create an Always Free Arm VM
- Compute → Instances → **Create instance**.
- **Image:** Canonical Ubuntu 24.04.
- **Shape:** change to **Ampere → VM.Standard.A1.Flex**, set **2 OCPUs / 12 GB RAM**
  (well within the Always Free allowance of 4 OCPU / 24 GB; plenty for the model).
- **Networking:** keep "Assign a public IPv4 address".
- **SSH keys:** upload your public key (or let Oracle generate and download one).
- Create. Note the instance's **public IP**.

> Arm (A1) capacity is sometimes exhausted in a region. If you get an
> "out of host capacity" error, wait and retry, or try a smaller shape (1 OCPU /
> 6 GB still works).

## 3. Open the firewall (two layers — both are required)
**a) Security list (Oracle side):**
- Networking → Virtual Cloud Networks → your VCN → its subnet → default
  Security List → **Add Ingress Rules**:
  - Source `0.0.0.0/0`, IP Protocol TCP, Destination port **80**
  - Source `0.0.0.0/0`, IP Protocol TCP, Destination port **443**

**b) Host firewall (Ubuntu side, run on the VM after you SSH in):** Oracle's
Ubuntu images ship an iptables rule that drops inbound traffic, so also run:
```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

## 4. Point a free domain at the VM
TLS certificates need a domain name (browsers won't do `wss://` to a bare IP).
- Go to https://www.duckdns.org, sign in, create a subdomain
  (e.g. `guitstrum`), and set its IP to your VM's **public IP**.
- Your recognizer URL will be `wss://guitstrum.duckdns.org`.

---

## 5. SSH in and install Docker + git-lfs
```bash
ssh ubuntu@<VM_PUBLIC_IP>

sudo apt-get update
sudo apt-get install -y git git-lfs
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
# log out and back in so the docker group applies:
exit
ssh ubuntu@<VM_PUBLIC_IP>
```

## 6. Clone the repo (with the LFS model)
```bash
git lfs install
git clone https://github.com/angelaayo/guitstrum-recognizer.git
cd guitstrum-recognizer
# confirm the model came down as real data, not a ~130-byte LFS pointer:
ls -lh chord_model.pkl     # should be ~165 MB
```

## 7. Configure the domain and start the stack
```bash
cp .env.prod.example .env
# edit .env so RECOGNIZER_DOMAIN matches your DuckDNS domain:
sed -i 's/your-subdomain.duckdns.org/guitstrum.duckdns.org/' .env

docker compose -f docker-compose.prod.yml up -d --build
```
The first build takes a few minutes (installs ffmpeg + the Python ML deps).
Caddy then requests a certificate automatically — give it ~30 seconds.

## 8. Verify
```bash
# from the VM or anywhere:
curl https://guitstrum.duckdns.org/health        # -> health JSON, valid TLS

docker compose -f docker-compose.prod.yml logs -f recognizer   # watch for the model load / predictions
```
Open `https://guitstrum.duckdns.org/health` in a browser — a valid padlock means
the certificate is live and `wss://` will work.

---

## 9. Wire it into the frontend
In the Vercel project (and local `.env`), set:
```
NEXT_PUBLIC_RECOGNIZER_WS_URL=wss://guitstrum.duckdns.org
```
Redeploy the frontend. The app connects to
`wss://guitstrum.duckdns.org/ws/recognize`.

## Updating later
```bash
cd guitstrum-recognizer
git pull
docker compose -f docker-compose.prod.yml up -d --build
```

## Notes
- The VM does not scale to zero — it runs 24/7, which is fine and free under
  Always Free. No cold starts, unlike the serverless options.
- CORS is already open (`allow_origins=["*"]`) in `main.py`.
