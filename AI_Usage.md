# AI Usage Across Pipeline Development

## Week 1

- No AI assistance was used during Week 1, as the work focused solely on image preprocessing, resizing, and data collection tasks.

## Week 2

- AI support was utilized to deepen understanding of the system components required for building the SfM pipeline:

  - Clarified which core modules and functions needed to be implemented for the full pipeline.
  - Asked questions regarding function behavior, arguments, and usage, such as the correct inputs for cv2.findEssentialMat().

- Used AI assistance to learn how to implement:
  - Triangulation
  - The cosine mask (for depth validation and error handling)
  - General geometric reasoning behind multi-view constraints

## Week 3

AI assistance became more essential as the pipeline grew in mathematical and computational complexity:

- Used AI to understand and implement Bundle Adjustment, especially:

  - How to optimize BA on a CPU-only system
  - How to construct and use a sparse Jacobian matrix for improved performance

- Used AI for additional clarification on OpenCV functions, their parameters, and expected outputs.

- Overall, AI acted as a technical reference to accelerate learning of complex optimization and geometry concepts.

- After the creation of the Week 3 notebook in the _Submission_Notebooks_ folder, AI was also used in order to breakdown the Week 3 notebook into different .py files for modularity and import them into a seperate notebook named _main.py_.
