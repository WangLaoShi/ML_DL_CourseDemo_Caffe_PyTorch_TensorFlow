# 使用 TensorFlow 来实现神经网络

![GAknhE](https://oss.images.shujudaka.com/uPic/GAknhE.png)

机器学习和深度学习都是人工智能的重要分支，它们之间的关系是深度学习是机器学习的一种，也可以说深度学习是机器学习的一个子集。机器学习主要通过数据训练模型，从而实现对新数据的预测或决策；而深度学习则是机器学习的一种实现方式，采用一种深度的神经网络来学习数据的表示，从而实现对新数据的预测或决策。深度学习框架是一种可以方便地构建、训练和优化深度神经网络的软件工具，可以帮助开发者更加高效地实现深度学习模型。

对于什么规模的问题需要用深度学习框架来解决，没有固定的标准答案。一般来说，深度学习框架适用于那些数据量非常大、模型非常复杂、需要处理复杂的非线性关系的问题。比如，在图像识别、自然语言处理、语音识别等领域，深度学习框架已经成为了主流的技术手段。但是，在一些规模较小、需要快速响应的问题中，使用传统的机器学习方法或者规则引擎可能会更加合适。因此，在实践中需要根据具体的问题规模、复杂程度、应用场景等因素来选择合适的方法和工具。

sklearn主要是一个机器学习库，它提供了很多传统机器学习算法的实现，比如线性回归、逻辑回归、KNN、SVM等。虽然神经网络也属于机器学习的一种，但与传统机器学习算法相比，神经网络更加复杂，参数更多，需要更高级的技术来训练和优化。因此，sklearn不支持神经网络可能是因为它更注重的是传统机器学习算法的实现和应用。

另外，sklearn虽然不支持神经网络，但是有很多其他的深度学习框架可以用来实现神经网络，比如TensorFlow、PyTorch、Keras等。这些框架提供了更加强大和灵活的神经网络模型、训练和优化算法，可以更好地适应各种不同的深度学习任务。因此，在使用时可以根据具体的需求和问题选择不同的框架和工具。

## 可视化

1. [TensorFlow Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.37038&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

2. [Visualizing Artificial Neural Networks (ANNs) with just One Line of Code](https://towardsdatascience.com/visualizing-artificial-neural-networks-anns-with-just-one-line-of-code-b4233607209e)

3. [5 Step Life-Cycle for Neural Network Models in Keras](https://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/)

![OFJbSN](https://oss.images.shujudaka.com/uPic/OFJbSN.jpg)

![qR0dH2](https://oss.images.shujudaka.com/uPic/qR0dH2.png)