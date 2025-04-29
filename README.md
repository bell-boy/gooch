# Gradient‑Oriented Operator Chain Harness (gooch)

## Build Instructions

To build the demo run 

```bash
make demos
```

To run the demo run

```bash
./build/demos/cs429submission.run n c i
```

where n = the number of samples , c = the number of classes, and i = the number of iterations performed

## Demo Explanation
Each iteration will be displayed in the following format: 
	The current loss using the cross entropy loss will be displayed on the first line, with the next lines displaying each sample's predicted and expected class for that iteration.
	The loss should steadily decrease to 0!
