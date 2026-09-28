// Domain Curriculum & Prerequisite DAG
// Subject: Machine Learning & Data Science Essentials

export const CURRICULUM_DOMAINS = [
  { id: "ml_fundamentals", title: "Machine Learning Essentials" },
  { id: "web_dev", title: "Full-Stack Web Engineering" },
  { id: "dsa", title: "Data Structures & Algorithms" }
];

export const CONCEPTS = {
  ml_fundamentals: [
    {
      id: "python_basics",
      title: "Python Data Structures & Syntax",
      domain: "ml_fundamentals",
      prerequisites: [],
      difficulty: 0.2,
      description: "Lists, dicts, control flow, functions, and modular code organization.",
      quiz: [
        {
          id: "q1",
          question: "Which data structure in Python guarantees unique elements and fast lookup?",
          options: ["List", "Tuple", "Set", "Dictionary Keys"],
          correctIndex: 2,
          explanation: "Sets store unique elements and use hash tables for O(1) average time complexity lookup."
        },
        {
          id: "q2",
          question: "What is the result of `list(filter(lambda x: x%2==0, [1,2,3,4]))`?",
          options: ["[1, 3]", "[2, 4]", "[True, False]", "[1, 2, 3, 4]"],
          correctIndex: 1,
          explanation: "Filter keeps items where condition returns True, so even numbers 2 and 4."
        }
      ],
      resources: {
        video: { title: "Python Crash Course for Data Science (15m)", url: "https://www.youtube.com/watch?v=rfscV0KH2wU", duration: "15 min" },
        text: { title: "Python Data Structures Guide", content: "Lists are ordered, mutable sequences. Sets are unordered collections of unique elements. Dicts map keys to values with O(1) lookup." },
        practice: { title: "Practice Problem: Frequency Counter", instruction: "Write a function that counts word frequencies in a sentence using a Python dictionary." }
      }
    },
    {
      id: "numpy_pandas",
      title: "NumPy & Pandas Data Wrangling",
      domain: "ml_fundamentals",
      prerequisites: ["python_basics"],
      difficulty: 0.4,
      description: "Vectorized arrays, DataFrames, filtering, aggregation, and handling missing data.",
      quiz: [
        {
          id: "q1",
          question: "How do you select rows where column 'age' is greater than 25 in Pandas?",
          options: ["df.select(df.age > 25)", "df[df['age'] > 25]", "df.filter('age > 25')", "df.loc(age > 25)"],
          correctIndex: 1,
          explanation: "Boolean indexing `df[df['age'] > 25]` creates a boolean mask and filters the DataFrame."
        },
        {
          id: "q2",
          question: "What is vectorized computation in NumPy?",
          options: ["Processing arrays with explicit Python loops", "Executing operations on whole arrays without explicit loops using C-optimized routines", "Running operations in parallel across threads", "Converting code to vector graphics"],
          correctIndex: 1,
          explanation: "Vectorization delegates looping to compiled C code, making array operations orders of magnitude faster."
        }
      ],
      resources: {
        video: { title: "Pandas & NumPy Matrix Mastery (20m)", url: "https://www.youtube.com/watch?v=vmEHCJofslg", duration: "20 min" },
        text: { title: "NumPy & Pandas Quick Reference", content: "NumPy arrays are homogeneous contiguous memory blocks. Pandas DataFrames add column labels and missing value management (NaN handling)." },
        practice: { title: "Practice Problem: Clean Customer CSV", instruction: "Fill missing age values with median age and drop rows with invalid emails using Pandas." }
      }
    },
    {
      id: "linear_algebra",
      title: "Linear Algebra & Matrix Operations",
      domain: "ml_fundamentals",
      prerequisites: ["python_basics"],
      difficulty: 0.5,
      description: "Vectors, matrix multiplication, dot products, eigenvalues, and transformations.",
      quiz: [
        {
          id: "q1",
          question: "If matrix A is size (3x2) and matrix B is size (2x4), what is the shape of dot product A @ B?",
          options: ["(3x4)", "(2x2)", "(3x2)", "Invalid operation"],
          correctIndex: 0,
          explanation: "Inner dimensions match (2==2), resulting matrix shape is (rows of A x cols of B) = (3x4)."
        },
        {
          id: "q2",
          question: "What does an Eigenvector represent during a linear transformation?",
          options: ["A vector whose direction reverses", "A direction that remains unchanged in orientation, scaled only by an eigenvalue", "A vector with zero length", "The orthogonal matrix transpose"],
          correctIndex: 1,
          explanation: "Ax = λx means matrix A only scales vector x along its original span without rotating it."
        }
      ],
      resources: {
        video: { title: "Essence of Linear Algebra - 3Blue1Brown (18m)", url: "https://www.youtube.com/watch?v=fNk_zzaMoSs", duration: "18 min" },
        text: { title: "Matrix Math for Machine Learning", content: "Dot products measure directional alignment. Matrix multiplications represent spatial linear transformations." },
        practice: { title: "Practice Problem: Matrix Projection", instruction: "Calculate the dot product and projection matrix of vector v onto u using NumPy." }
      }
    },
    {
      id: "gradient_descent",
      title: "Calculus & Gradient Descent",
      domain: "ml_fundamentals",
      prerequisites: ["linear_algebra"],
      difficulty: 0.6,
      description: "Partial derivatives, loss functions, learning rates, and optimization pathways.",
      quiz: [
        {
          id: "q1",
          question: "What happens if your Learning Rate (α) is set too high during gradient descent?",
          options: ["Slow convergence", "Oscillation or divergence away from optimal minimum", "The gradient becomes zero instantly", "Loss drops to zero in one step"],
          correctIndex: 1,
          explanation: "Too large learning rate causes overshooting of parameter updates across local minima."
        },
        {
          id: "q2",
          question: "The gradient vector ∇f points in which direction?",
          options: ["Direction of steepest decrease", "Direction of steepest increase of the function", "Orthogonal to contour lines always downwards", "Parallel to axis"],
          correctIndex: 1,
          explanation: "The gradient points towards maximum rate of increase. In optimization we move opposite (-∇f)."
        }
      ],
      resources: {
        video: { title: "Gradient Descent Visual Intuition (12m)", url: "https://www.youtube.com/watch?v=IHZwWFHWa-w", duration: "12 min" },
        text: { title: "Optimization & Convex Loss Functions", content: "Gradient descent iteratively updates θ = θ - α * ∇L(θ) to reach minimal loss value." },
        practice: { title: "Practice Problem: Code 1D Gradient Descent", instruction: "Implement derivative calculation for f(x) = x^2 + 4x + 4 and find x minimum with α=0.1." }
      }
    },
    {
      id: "linear_regression",
      title: "Linear & Logistic Regression",
      domain: "ml_fundamentals",
      prerequisites: ["numpy_pandas", "gradient_descent"],
      difficulty: 0.65,
      description: "Supervised prediction, cost function MSE, Sigmoid activation, and binary classification.",
      quiz: [
        {
          id: "q1",
          question: "What function maps linear output to probability (0 to 1) in Logistic Regression?",
          options: ["ReLU", "Sigmoid σ(z) = 1 / (1 + e^-z)", "Softmax", "Tanh"],
          correctIndex: 1,
          explanation: "The sigmoid activation function transforms real numbers into valid probabilities between 0 and 1."
        },
        {
          id: "q2",
          question: "Which metric is commonly minimized for Linear Regression?",
          options: ["Cross-Entropy Loss", "Mean Squared Error (MSE)", "Hinge Loss", "Accuracy"],
          correctIndex: 1,
          explanation: "MSE measures average squared difference between predictions and actual targets."
        }
      ],
      resources: {
        video: { title: "Linear & Logistic Regression Explained (16m)", url: "https://www.youtube.com/watch?v=yIYKR4sgzI8", duration: "16 min" },
        text: { title: "Supervised Learning Fundamentals", content: "Linear regression predicts continuous numeric values, while Logistic regression outputs class membership probabilities." },
        practice: { title: "Practice Problem: Predict House Prices", instruction: "Train Scikit-Learn LinearRegression on housing features and evaluate RMSE." }
      }
    },
    {
      id: "neural_networks",
      title: "Deep Learning & Neural Networks",
      domain: "ml_fundamentals",
      prerequisites: ["linear_regression"],
      difficulty: 0.85,
      description: "Multilayer Perceptrons, backpropagation, activation functions, and PyTorch basics.",
      quiz: [
        {
          id: "q1",
          question: "What role does non-linear activation (like ReLU or Sigmoid) play in a neural network?",
          options: ["Speeds up network memory copy", "Allows the network to learn complex non-linear boundary decision functions", "Prevents weights from becoming negative", "Ensures batch normalization"],
          correctIndex: 1,
          explanation: "Without non-linear activations, stacking multiple linear layers collapses into a single linear transformation."
        },
        {
          id: "q2",
          question: "What algorithm computes gradients for hidden layers using the calculus chain rule?",
          options: ["Forward Pass", "Backpropagation", "K-Means Clustering", "Principal Component Analysis"],
          correctIndex: 1,
          explanation: "Backprop propagates error backward through the computational graph to compute partial derivatives."
        }
      ],
      resources: {
        video: { title: "Neural Networks & Backpropagation (22m)", url: "https://www.youtube.com/watch?v=aircAruvnKk", duration: "22 min" },
        text: { title: "Deep Learning Architecture Guide", content: "Layers consist of neurons performing w*x + b followed by non-linear activation functions." },
        practice: { title: "Practice Problem: PyTorch Classifier", instruction: "Construct a 2-layer MLP in PyTorch to classify 2D synthetic concentric circle dataset." }
      }
    }
  ]
};
