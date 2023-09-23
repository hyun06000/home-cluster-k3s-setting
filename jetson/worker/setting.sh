# uninstall existing k3s
/usr/local/bin/k3s-uninstall.sh

# requirements
sudo apt-get update
sudo apt-get install -y curl

# install k3s
read -p "master node ip : " YOUR_SERVER_NODE_IP
read -p "master node toekn : " YOUR_NODE_TOKEN
curl -sfL https://get.k3s.io | K3S_URL=https://${YOUR_SERVER_NODE_IP}:6443 K3S_TOKEN=${YOUR_NODE_TOKEN} sh -

# check status
sudo systemctl status k3s
