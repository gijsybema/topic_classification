echo "Install Python packages"
pip3 install --index-url https://repo.eap.aon.com/artifactory/api/pypi/pypi/simple --upgrade -r requirements.txt --trusted-host repo.eap.aon.com 
