# The emergence of cooperation by evolutionary generalization

Python code for the simulations in the paper "The emergence of cooperation by evolutionary generalization", published in the Proc. Royal Soc. B, 2021, by Félix Geoffroy and Jean-Baptiste André.

To run a simulation, in a terminal, and in the `code/` subfolder, type

```bash
$ python main.py
```

The resulting text files are saved in the `results/` subfolder. The two `metrics` files record the performance of players during the pre-selection and the selection phase. Additionally, the first two lines of these files contain all parameters used in the simulation. The two `totalSet` files contain the behavior of both players (on both the training and a test set) at the end of the pre-selection and the selection phase. Lastly, one can obtain a `heatmap` file by choosing `HeatMapFile = True` in `Parameters.py`. This file was used to produce Figure 5 of the article.

If more details are needed on the code, please contact the first author of the publication at felix.geoffroy.fr@gmail.com.
