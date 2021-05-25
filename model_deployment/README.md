## Step to run this project:

> NOTE: To run this project you will need Graphics Card (GPU).
> To run on CPU please checkout here.

#### 1. Clone this repo or download the zip file.
#### 2. Ensure you have Python3 and pip already installed on your system.
#### 3. Navigate to VIRTUON/model_deployment
#### 4.1 Setup virtual environment in Ubuntu
 - Run the following commands in terminal
 ```
    pip3 install virtualenv 
    virtualenv venv
    source venv/bin/activate
 ```
 
#### 4.2 Setup virtual environment in Windows in CMD
 - Run the following commands in cmd.exe
 ```
    python3 -m pip3 install virtualenv
    python3 -m virtualenv venv
    venv\Scripts\activate.bat
 ```
 
 #### 4.3 Setup virtual environment in Windows in PowerShell
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
 
#### 5. Install the dependencies
 - Note you should be in model_deployment folder and ensure requirements.txt exist.
 - Run the following command
 ``` 
  pip3 install -r requirements.txt
 ```
 - Note: This will take a while. It will be of approx. 850MB

#### 6. Starting the server
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
   
