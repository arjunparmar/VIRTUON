## Step to run this project:

- <b>[METHOD 1: Setting up Environment](#method-1)</b>
- <b>[METHOD 2: Running Docker Image](#method-2)</b>

### Method 1:

1. Clone this repo or download the zip file.
2. Ensure you have Python3 and pip already installed on your system.
3. Navigate to VIRTUON/model_deployment
4. Setup virtual environment 
- In Linux
  - Run the following commands in terminal
 ```
    pip3 install virtualenv 
    virtualenv venv
    source venv/bin/activate
 ```
 
- In Windows CMD
  - Run the following commands in cmd.exe
 ```
    python3 -m pip3 install virtualenv
    python3 -m virtualenv venv
    venv\Scripts\activate.bat
 ```
 
+ In Windows PowerShell
   - Run the following commands in PowerShell.exe
 > Note: By default, external script execution is disabled in Powershell.</br> For more details refer [here](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.1).

 ```
    python3 -m pip3 install virtualenv 
    python3 -m virtualenv venv
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    venv\Scripts\Activate.ps1
 ```
 > Note : To deactivate the Virtual Enviorment use `deactivate` command.</br>
 For more detials about Virtual Enviroments in Python checkout [this.](https://docs.python.org/3/library/venv.html)
 
5. Install the dependencies
 - Note you should be in model_deployment folder and ensure requirements.txt exist.
 - Run the following command
 ``` 
  pip3 install -r requirements.txt
 ```
 > Note: This will take a while. It will be of approx. 850MB

6. Starting the server
 - When starting it for first time, run the following command.
```
  python3 manage.py migrate
```
 - Now run
```
  python3 manage.py runserver
```
 - Copy and paste this address in your browser
```
  http://127.0.0.1:8000/
``` 
 - Tryon :)

### Method 2

1. Ensure you have Docker installed properly. If not refer [this](https://docs.docker.com/engine/install/).
2. Run the following command to pull the image. [Link](https://hub.docker.com/r/pra17dod/virtuon) to the Docker image.
```
docker pull pra17dod/virtuon
```
3. After successfully pulling the image. Run the image.
```
docker run -it --name virtuon -p 8000:8000 pra17dod/virtuon
```
 - Your server is up and running. To know more about [Docker](https://docs.docker.com/get-started/overview/).
 - Copy and paste this address in your browser
```
  http://0.0.0.0:8000/
```
 - Tryon :)

### For any installation problem or queries feel free to create an issue :)
