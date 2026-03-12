from pylab import *
import os


def create_custom_file(name, data):
    with open(name, 'w') as file:
        if isinstance(data, str):
            file.write(data)
        if isinstance(data, dict):
            for k, v in sorted(data.items()):
                line = str(k) + ": " + str(v) + "\n"
                file.write(line)


def merge_configs(cfg1, cfg2):
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def save_lr_distribution(learning_rates, path_to_save):
    epochs = list(range(1, len(learning_rates) + 1))

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, marker='o', linestyle='-', color='b')

    # Etiquetas y título
    plt.title('Variación del Learning Rate durante el Entrenamiento')
    plt.xlabel('Iteraciones')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(os.path.join(path_to_save, "learning_rate_variation.png"))
    # plt.show()
    plt.clf()


def create_dir(ruta_directorio):
    try:
        if not os.path.exists(ruta_directorio):
            os.makedirs(ruta_directorio)
        else:
            return f"El directorio ya existe: {ruta_directorio}"
    except Exception as e:
        return f"Error al crear el directorio: {str(e)}"