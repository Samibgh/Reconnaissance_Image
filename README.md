# Reconnaissance Visage Voix

* [English](#English)
* [Français](#Français)


# English
## Dependencies

Running the application can be done following the instructions above:

If you're a conda user, you can create an environment from the ```environment.yml``` file using the Terminal or an Anaconda Prompt for the following steps:

1. Create the environment from the ```environment.yml``` file:

    ```conda env create --name facerecognition -f environment.yml```
   * Make sure you are in the place where you cloned the github.
    
2. Activate the new environment:
    
    ```conda activate facerecognition```

3. Verify that the new environment was installed correctly:

    ```conda list```
    
You can also clone the environment through the environment manager of Anaconda Navigator.

## Use

Within the virtual environment:

```streamlit run app.py```

A web application will open in the prompted URL. The options there are:
* checkbox for voice command: say "tarte"
* button to turn on the cam with a click
* button on the right to stop the cam

At the end, if the command "ctrl-c" in your terminal doesn't stop the warning, kill the terminal.

# Français

## Dépendances
L'exécution de l'application peut être effectuée en suivant les instructions ci-dessous :

Si vous êtes un utilisateur de conda, vous pouvez créer un environnement à partir du fichier environment.yml en utilisant le Terminal ou un Anaconda Prompt pour les étapes suivantes :

Créez l'environnement à partir du fichier ```environment.yml``` :

```conda env create --name facerecognition -f environment.yml```

Activez le nouvel environnement :

```conda activate facerecognition```

Vérifiez que le nouvel environnement a été installé correctement :

```conda list```

Vous pouvez également cloner l'environnement via le gestionnaire d'environnement d'Anaconda Navigator.

## Utilisation
Dans l'environnement virtuel :

```streamlit run app.py```

Une application web s'ouvrira à l'URL indiquée. Les options disponibles sont :
* case à cocher pour commade vocal : dire "tarte"
* bouton pour allumer la cam avec un clique
* bouton à droit pour arrêter la cam


à la fin de votre utilisation, si le ctrl-c sur votre terminal n'arrête pas le message alors kill le terminal.
