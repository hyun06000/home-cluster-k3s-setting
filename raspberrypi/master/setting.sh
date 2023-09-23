# uninstall existing k3s
/usr/local/bin/k3s-uninstall.sh


# install requirements
sudo apt-get update
sudo apt-get install -y curl

# install k3s
curl -sfL https://get.k3s.io | sh -

# get nodes
sudo k3s kubectl get nodes
