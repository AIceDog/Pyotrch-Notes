# Test
Translator test.
layer = nn.Conv2d(1, 3, kernel_size = 3, stride = 1, padding = 0)
###  参数解读:
###  1是输入图片的channel, 比如[b(图片数量), 1(图片channel是1, 表示该图片为灰度图), 28(图片的宽), 28(图片的高)];
###  3是输出图片的channel, 输出图片的channel是3, 输出图片的dimension是[b(图片数量), 3, new width, new hight];
###  kernel_size是kernel的宽和高
###  stride是步长
###  如果padding是0, 输入图片的宽高不变, 如果padding是1, 输入图片的宽高向外扩张1像素

x = torch.rand(5, 1, 28, 28) # 设置输入图片, 输入图片数量是5, channel是1, 宽和高都是28

out = layer.forward(x) #完成一次卷积的前向运算, 输入图片是x
### output 是 torch.Size([5, 3, 26, 26)]
