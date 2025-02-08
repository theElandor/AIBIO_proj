#!/bin/bash
#SBATCH -e /homes/nmorelli/output/err.txt
#SBATCH -o /homes/nmorelli/output/out.txt
#SBATCH --job-name=simclr_train
#SBATCH --account=ai4bio2024
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu_RTXA5000_24G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_A40_48G"


JOB_CMD="python3 /homes/nmorelli/AIBIO_proj/source/train_backbone.py /homes/nmorelli/AIBIO_proj/config/train/general_conf.yaml"


# Replace with your actual Discord webhook URL
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1337444251608285305/_wIi-3jEM_RGpdThArOC7einausDxB-BD41Qd4DuZkLSJlw4odF84RZLmmKmUKHfPPj8"

##############################
# Definizione del comando da eseguire
##############################
# Utilizziamo apici doppi all'esterno e scappiamo le virgolette interne
export JOB_CMD

##############################
# Helper functions
##############################

# Send a plain text notification (for job start)
function send_discord_notification {
  local message="$1"
  # Use Python to JSON-encode the message
  encoded_message=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$message")
  payload="{\"content\": $encoded_message}"
  
  curl -H "Content-Type: application/json" \
       -X POST \
       -d "$payload" \
       "$DISCORD_WEBHOOK_URL"
}

# Send an embed message using Python for payload generation (for job finish)
function send_discord_embed {
  # Generate the embed payload via Python
  embed_payload=$(python3 <<'EOF'
import json, os

# Recupera le variabili d'ambiente
job_id    = os.environ.get("SLURM_JOB_ID", "Unknown")
node      = os.environ.get("SLURM_NODELIST", "Unknown")
job_state = os.environ.get("SLURM_JOB_STATE", "Unknown")
job_name  = os.environ.get("SLURM_JOB_NAME", "Unknown")
job_cmd   = os.environ.get("JOB_CMD", "Not available")

# Legge fino a 800 caratteri dai file di log
def read_log(filepath):
    try:
        with open(filepath, "r") as f:
            return f.read(800)
    except Exception as e:
        return f"Error reading {filepath}: {e}"

out_log = read_log("/homes/nmorelli/output/out.txt")
err_log = read_log("/homes/nmorelli/output/err.txt")

# Imposta il colore dell'embed: verde se COMPLETED, rosso altrimenti
embed_color = 3066993 if job_state == "COMPLETED" else 15158332

embed = {
    "embeds": [{
        "title": "SLURM Job Finished",
        "description": f"Job **{job_id}** (name: **{job_name}**) on node **{node}** finished with state **{job_state}**.",
        "color": embed_color,
        "fields": [
            {"name": "Command Executed", "value": f"```{job_cmd}```", "inline": False},
            {"name": "Output Log (first 800 characters)", "value": f"```{out_log}```", "inline": False},
            {"name": "Error Log (first 800 characters)", "value": f"```{err_log}```", "inline": False}
        ],
        "footer": {"text": "SLURM Job Notification"}
    }]
}

print(json.dumps(embed))
EOF
)
  
  curl -H "Content-Type: application/json" \
       -X POST \
       -d "$embed_payload" \
       "$DISCORD_WEBHOOK_URL"
}

##############################
# Notifications and Job Command
##############################

# Notifica di inizio job includendo job name e comando eseguito
send_discord_notification "⚙️ **SLURM Job $SLURM_JOB_ID ($SLURM_JOB_NAME)** started on node *$SLURM_NODELIST*. 
⚙️Executing command: \`$JOB_CMD\`"

# Esecuzione del comando definito
eval "$JOB_CMD"

# Attende un attimo per garantire che i log vengano scritti
sleep 1

# Notifica di fine job con embed (includendo anche job name e comando eseguito)
send_discord_embed