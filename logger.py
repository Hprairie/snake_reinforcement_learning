import json
import sys
import matplotlib.pyplot as plt


def save_training_logs(model_logs, version):
    with open('model_logs/{:s}.txt'.format(version), 'w') as file:
        file.write(json.dumps(model_logs, indent=4))


def display_training_logs(model_logs, version):
    # Plot score
    plt.plot(model_logs['game'],
             model_logs['score'],
             label='Score of {} using {}'.format(model_logs['network_name'],
                                                 version))

    # Plot mean score
    plt.plot(model_logs['game'],
             model_logs['mean_score'],
             label='Mean of {} using {}'.format(model_logs['network_name'],
                                                version))


if __name__ == "__main__":
    plt.figure(figsize=(10, 5))

    for version in sys.argv[1:]:
        with open('model_logs/{:s}.txt'.format(version), 'r') as f:
            model_logs = json.load(f)

            # Display all the logged information to screen
            display_training_logs(model_logs, version)

    plt.xlabel('Games Played')
    plt.ylabel('Score')
    plt.title('Score and Mean Score over Games Played')
    plt.legend()
    plt.show()
