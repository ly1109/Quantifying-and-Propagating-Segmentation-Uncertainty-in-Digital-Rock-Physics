# config.py

metadata = {
    "Berea": {
        "number": 1,
        "original": "./Digital Rock/Data/Grayscale/Berea_2d25um_grayscale_filtered.raw",
        "segmented": "./Digital Rock/Data/Binary/Berea_2d25um_binary.raw",
        "seed": 42
    },
    "BanderaBrown": {
        "number": 2,
        "original": "./Digital Rock/Data/Grayscale/BanderaBrown_2d25um_grayscale_filtered.raw",
        "segmented": "./Digital Rock/Data/Binary/BanderaBrown_2d25um_binary.raw",
        "seed": 43
    },
    "BanderaGray": {
        "number": 3,
        "original": "./Digital Rock/Data/Grayscale/BanderaGray_2d25um_grayscale_filtered.raw",
        "segmented": "./Digital Rock/Data/Binary/BanderaGray_2d25um_binary.raw",
        "seed": 44
    },
    "Bentheimer": {
        "number": 4,
        "original": "./Digital Rock/Data/Grayscale/Bentheimer_2d25um_grayscale_filtered.raw",
        "segmented": "./Digital Rock/Data/Binary/Bentheimer_2d25um_binary.raw",
        "seed": 45
    },
    "BSG": {
        "number": 5,
        "original": "./Digital Rock/Data/Grayscale/BSG_2d25um_grayscale_filtered.raw",
        "segmented": "./Digital Rock/Data/Binary/BSG_2d25um_binary.raw",
        "seed": 46
    },
    "BUG": {
        "number": 6,
        "original": "./Digital Rock/Data/Grayscale/BUG_2d25um_grayscale_filtered.raw",
        "segmented": "./Digital Rock/Data/Binary/BUG_2d25um_binary.raw",
        "seed": 47
    },
    "BB": {
        "number": 7,
        "original": "./Digital Rock/Data/Grayscale/BB_2d25um_grayscale_filtered.raw",
        "segmented": "./Digital Rock/Data/Binary/BB_2d25um_binary.raw",
        "seed": 48
    },
    "CastleGate": {
        "number": 8,
        "original": "./Digital Rock/Data/Grayscale/CastleGate_2d25um_grayscale_filtered.raw",
        "segmented": "./Digital Rock/Data/Binary/CastleGate_2d25um_binary.raw",
        "seed": 49
    },
    "Kirby": {
        "number": 9,
        "original": "./Digital Rock/Data/Grayscale/Kirby_2d25um_grayscale_filtered.raw",
        "segmented": "./Digital Rock/Data/Binary/Kirby_2d25um_binary.raw",
        "seed": 50
    },
    "Leopard": {
        "number": 10,
        "original": "./Digital Rock/Data/Grayscale/Leopard_2d25um_grayscale_filtered.raw",
        "segmented": "./Digital Rock/Data/Binary/Leopard_2d25um_binary.raw",
        "seed": 51
    },
    "Parker": {
        "number": 11,
        "original": "./Digital Rock/Data/Grayscale/Parker_2d25um_grayscale_filtered.raw",
        "segmented": "./Digital Rock/Data/Binary/Parker_2d25um_binary.raw",
        "seed": 52
    }
}

available_images = list(metadata.keys())

dimensions = {"width": 1000,
              "height": 1000,
              "depth": 1000}

data_directory = {"train": "./dataset/train",
                  "test": "./dataset/test",
                  "folders": ['images', 'masks']}
