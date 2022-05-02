# Test
Translator test.
layer = nn.Conv2d(1, 3, kernel_size = 3, stride = 1, padding = 0) # 这是创建了一个类
### 如果写成 F.conv2d(...) 则表示调用了函数, pyotrch 中开头小写表示调用函数，开头大写表示调用类
###  参数解读:
###  1是输入图片的channel, 比如[b(图片数量), 1(图片channel是1, 表示该图片为灰度图), 28(图片的宽), 28(图片的高)]
###  3是输出图片的channel, 输出图片的channel是3, 输出图片的dimension是[b(图片数量), 3, new width, new hight]
###  kernel_size是kernel的宽和高
###  stride是步长
###  如果padding是0, 输入图片的宽高不变, 如果padding是1, 输入图片的宽高向外扩张1像素

x = torch.rand(5, 1, 28, 28) # 设置输入图片, 输入图片数量是5, channel是1, 宽和高都是28

out = layer.forward(x) # 完成一次卷积的前向运算, 输入图片是x
### output 是 torch.Size([5, 3, 26, 26)]

out = layer(x)  # 完成一次卷积的前向运算, 不过会先运行 hooks, 再运行 .forward 函数

layer.weight # 输出类layer的成员变量weight, 如果require_grad = True, 则在backpropag过程中 layer.weight 会自动更新

layer.weight.shape

layer.bias.shape




###  layer = nn.Conv2d(3, 16, kernel_size = 5, stride = 1, padding = 0) 的另一种写法
x = torch.rand(5, 3, 28, 28) # 输入图片x, [5(图片数量), 3(图片channel是3, 表示该图片为rgb图), 28(图片的宽), 28(图片的高)]
w = torch.rand(16, 3, 5, 5) # layer.weight
b = torch.rand(16) # layer.bias

out = F.conv2d(x, w, b, stride = 1, padding = 1)

