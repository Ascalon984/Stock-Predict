try:
    import tensorflow as tf
    print("TF Imported successfully")
    print(f"Version: {tf.__version__}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Other Error: {e}")

try:
    from tensorflow import keras
    print("Keras Imported from TF")
except Exception as e:
    print(f"Keras Import Error: {e}")
