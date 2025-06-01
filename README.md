# 基于TensorFlow Lite实现的Android花卉识别应用详细分析与实现方案
## 一、应用概述
本Android花卉识别应用基于TensorFlow Lite框架构建
## 实现步骤
准备工作
 **下载初始代码**：1.先从https://github.com/hoitab/TFLClassify下载项目文件并且解压到Android studio项目文件中。
1. 选择模块
2. 导入模型
3. 修改一下配置文件中tensorflow的版本。
4. 完成TODO
   private class ImageAnalyzer(ctx: Context, private val listener: RecognitionListener) :
   ImageAnalysis.Analyzer {

        // TODO 1: Add class variable TensorFlow Lite Model
        // Initializing the flowerModel by lazy so that it runs in the same thread when the process
        // method is called.
        private val flowerModel: FlowerModel by lazy{

            // TODO 6. Optional GPU acceleration
            val compatList = CompatibilityList()

            val options = if(compatList.isDelegateSupportedOnThisDevice) {
                Log.d(TAG, "This device is GPU Compatible ")
                Model.Options.Builder().setDevice(Model.Device.GPU).build()
            } else {
                Log.d(TAG, "This device is GPU Incompatible ")
                Model.Options.Builder().setNumThreads(4).build()
            }

            // Initialize the Flower Model
            FlowerModel.newInstance(ctx, options)
        }

        override fun analyze(imageProxy: ImageProxy) {

            val items = mutableListOf<Recognition>()

            // TODO 2: Convert Image to Bitmap then to TensorImage
            val tfImage = TensorImage.fromBitmap(toBitmap(imageProxy))

            // TODO 3: Process the image using the trained model, sort and pick out the top results
            val outputs = flowerModel.process(tfImage)
                .probabilityAsCategoryList.apply {
                    sortByDescending { it.score } // Sort with highest confidence first
                }.take(MAX_RESULT_DISPLAY) // take the top results

            // TODO 4: Converting the top probability items into a list of recognitions
            for (output in outputs) {
                items.add(Recognition(output.label, output.score))
            }

//            // START - Placeholder code at the start of the codelab. Comment this block of code out.
//            for (i in 0 until MAX_RESULT_DISPLAY){
//                items.add(Recognition("Fake label $i", Random.nextFloat()))
//            }
//            // END - Placeholder code at the start of the codelab. Comment this block of code out.

            // Return the result
            listener(items.toList())

            // Close the image,this tells CameraX to feed the next image to the analyzer
            imageProxy.close()
        }

    // TODO 5: Optional GPU Delegates
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.3.0'

最后实现效果：
[]()
![Screenshot_20250601_210301_TFL Classify.jpg](Screenshot_20250601_210301_TFL%20Classify.jpg)
![Screenshot_20250601_210323_TFL Classify.jpg](Screenshot_20250601_210323_TFL%20Classify.jpg)
![Screenshot_20250601_210349_TFL Classify.jpg](Screenshot_20250601_210349_TFL%20Classify.jpg)

