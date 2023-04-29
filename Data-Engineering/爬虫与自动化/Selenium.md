# Selenium

## 安装

```powershell
$ pip install selenium
```



## 驱动设置

```python
from selenium.webdriver.chrome.service import Service
from selenium import webdriver

chrome_service = Service('./chrome_webdriver/chromedriver')
driver = webdriver.Chrome(service=chrome_service)

```



## 元素

### 元素定位

定位八大元素定位方式id、name、class\_name、tag\_name、link\_text、partial\_link\_text、xpath、css\_selector

查找元素使用webdriver对象下的`find_element()`方法，需要导入 `By`来确定查询方式

```python
from selenium.webdriver.common.by import By
```

#### ID

一般情况下id在当前页面中是唯一的，当元素存在id属性值时，优先使用id方式定位元素。

```html
<input type="text" class="s_ipt" name="wd" id="kw" maxlength="100" autocomplete="off">
```

```python
driver.find_element(By.ID, "kw")
```

#### NAME

元素的name属性值可能存在重复，当页面内有多个元素的特征值相同时，定位元素的方法执行时只会默认获取第一个符合要求的特征对应的元素。

```html
<input type="text" class="s_ipt" name="wd" id="kw" maxlength="100" autocomplete="off">
```

```python
driver.find_element(By.NAME, "wd")
```

#### CLASS\_NAME

一个class可能会存在多个值（每个属性值以空格隔开），只能使用其中的任意一个属性值进行定位。

```html
<input class="bg s_btn btn_h btnhover" type="text" name="key">
```

```python
driver.find_element(By.CLASS_NAME,"bg")
```

#### TAG\_NAME

通过元素的标签名称来定位，例如input标签、button标签、a标签等。

```python
driver.find_element(By.TAG_NAME, "input")

```

#### LINK\_TEXT & PARTIAL\_LINK\_TEXT

LINK\_TEXT**  **和 PARTIAL\_LINK\_TEXT定位超链接标签。只针对超链接元素（a 标签）

LINK\_TEXT只能使用精准匹配（需要输入超链接的全部文本信息）

PARTIAL\_LINK\_TEXT可以使用精准或模糊匹配

```html
<a href="http://XXX">联系客服</a>
```

```python
driver.find_element(By.LINK_TEXT, "联系客服")
driver.find_element(By.PARTIAL_LINK_TEXT, "联系")

```

#### XPATH

**XPATH**(XML Path Language)，用于解析XML和HTML(XML多用于传输和存储数据，侧重于数据，HTML多用于显示数据并关注数据的外观)。使用浏览器开发者工具直接复制xpath路径值。

**1. 绝对路径**

以/html开始，使用 / 来分割元素层级的语法，由多个同级标签的用索引区别，下标是从1开始，还可以用last()、position()函数来表达索引

```python
driver.find_element(By.XPATH, "/html/body/div[2]/div/div[2]/div[1]/form/input[1]")
```

**2. 相对路径**

相对路径是以 // 开始， // 后面跟元素名称，不知元素名称时可以使用 \* 号代替

```html
<input class="but1" type="text" name="key" placeholder="请输入你要查找的关键字" value="">
<a href="http://127.0.0.1/register">免费注册</a>

```

```python
# 单个属性
driver.find_element(By.XPATH, "//标签名[@属性='属性值']")
driver.find_element(By.XPATH, "//*[@属性='属性值']")
# 多个属性
driver.find_element(By.XPATH, "//标签名[@属性1='属性值1' and @属性2='属性值2']")
driver.find_element(By.XPATH, "//*[@属性1='属性值1' and @属性2='属性值2']")
# 属性模糊匹配
driver.find_element(By.XPATH, "//标签名[contains(@属性,'属性值的部分内容')]")
driver.find_element(By.XPATH, "//*[contains(@属性,'属性值的部分内容')]")
# starts-with 属性值开头匹配
driver.find_element(By.XPATH, "//标签名[starts-with(@属性,'属性值的开头部分')]")
driver.find_element(By.XPATH, "//*[starts-with(@属性,'属性值的开头部分')]")
# text() 文本值定位
# 通过标签的文本值进行定位，定位文本值等于XX的元素，一般适用于p标签、a标签
driver.find_element(By.XPATH, "//标签名[text()='文本信息']")
driver.find_element(By.XPATH, "//*[text()='文本信息']")


# example
driver.find_element(By.XPATH, "//input")
driver.find_element(By.XPATH, "//input[@class='but1' and @placeholder='请输入你要查找的关键字']")
driver.find_element(By.XPATH, "//input[contains(@placeholder,'查找')]")
driver.find_element(By.XPATH, "//input[starts-with(@placeholder,'请输入')]")
driver.find_element(By.XPATH, "//*[text()='免费注册']")

```

#### CSS\_SELECTOR

CSS定位效率高于XPATH，使用浏览器开发者工具直接复制selector路径值。

**1. 绝对路径**

以html开始，使用 > 或 空格 分隔，与XPATH一样，CSS\_SELECTOR的下标也是从1开始

```python
driver.find_element(By.CSS_SELECTOR, "html>body>div>div>div>div>form>input:nth-child(1)")
```

**2. 相对路径**

不以html开头，以CSS选择器开头，比如标id选择器、class选择器等

```python
<html>
  <body>
    <form id="loginForm">
      <input class="required" name="username" type="text" />
      <input class="required passfield" name="password" type="password" />
      <input name="continue" type="submit" value="Login" />
      <input name="continue" type="button" value="Clear" />
    </form>
  </body>
<html>

```

```python
# id选择器
driver.find_element(By.CSS_SELECTOR, "标签#id属性值")
# class选择器
driver.find_element(By.CSS_SELECTOR, ".class属性值")
driver.find_element(By.CSS_SELECTOR, "[class='class属性值']")
# 属性选择器
driver.find_element(By.CSS_SELECTOR, "标签名[属性='属性值']")
driver.find_element(By.CSS_SELECTOR, "[属性='属性值']")
driver.find_element(By.CSS_SELECTOR, "标签名[属性1='属性值1'][属性2='属性值2']")
# 模糊匹配
driver.find_element(By.CSS_SELECTOR, "[属性^='开头的字母']") # 获取指定属性以指定字母开头的元素
driver.find_element(By.CSS_SELECTOR, "[属性$='结束的字母']") # 获取指定属性以指定字母结束的元素
driver.find_element(By.CSS_SELECTOR, "[属性*='包含的字母']") # 获取指定属性包含指定字母的元素
# 标签选择器
driver.find_element(By.CSS_SELECTOR, "标签名")


# example
# id选择器
driver.find_element(By.CSS_SELECTOR, "form#loginForm")
# class选择器
driver.find_element(By.CSS_SELECTOR, ".required")
driver.find_element(By.CSS_SELECTOR, "[class='required']")
# 属性选择器
driver.find_element(By.CSS_SELECTOR, "input[name='username']")
driver.find_element(By.CSS_SELECTOR, "[name='username']")
driver.find_element(By.CSS_SELECTOR, "input[name='password'][type='password']")
# 模糊匹配
driver.find_element(By.CSS_SELECTOR, "[name^='pas']")
# 标签选择器
driver.find_element(By.CSS_SELECTOR, "input").send_keys('123')

```

**3. 层级关系**

1.  first-child：第一个元素
2.  last-child：最后一个子元素
3.  nth-child( )：正序，下标从1开始
4.  nth-last-child( )：倒序，下标从1开始

注：若一个标签下有多个同级标签，虽然这些同级标签的 tag\_name 不一样，但是他们是放在一起排序的

```html
<div class="help">
  <a href="https://www.baidu.com/">百度首页</a>
  <a href="https://news.baidu.com/">百度新闻</a>
  <a href="https://image.baidu.com/">百度图片</a>
</div>
```

```python
driver.find_element(By.CSS_SELECTOR, ".help>a:first-child").click() # 百度首页
driver.find_element(By.CSS_SELECTOR, ".help>a:last-child").click() # 百度图片
driver.find_element(By.CSS_SELECTOR, ".help>a:nth-child(2)").click()  # 百度新闻
driver.find_element(By.CSS_SELECTOR, ".help>a:nth-last-child(2)").click()  # 百度新闻
```

### 元素交互

#### 点击click

```python
driver.find_element(By.ID, "su").click()
```

#### 输入send\_keys

文件上传也可以通过send\_keys完成

```python
driver.find_element(By.ID, "kw").send_keys("chat-gpt")
driver.find_element(By.TAG_NAME, "input").send_keys("./data.txt")

```

#### 清除内容clear

```python
driver.find_element(By.ID, "kw").clear()
```

## 等待方式

#### sleep

```python
import time
time.sleep(10)
```

#### implicitly\_wait

隐式等待，就是在设置的等待时间范围内**全局等待**，只要全部加载完就会立即结束等待继续执行。如果超时，则抛出异常。

一旦设置，那么这个等待在浏览器对象的整个生命周期起作用

```python
driver.implicitly_wait(10) 
chrome_driver.get("http://www.baidu.com")
chrome_driver.find_element_by_xpath('//*[@id="s-top-loginbtn"]').click()
chrome_driver.find_element_by_xpath('//*[@id="s-top-loginbtn1"]').click()

```

#### WebDriverWait

显示等待，要明确等待条件和等待上限。创建WebDriverWait()对象后，要结合until( )或者until\_not( ) 和expected\_conditions一起使用

1.  driver：浏览器驱动 &#x20;
2.  timeout：等待上限，单位是秒 &#x20;
3.  poll\_frequency：检测的轮询间隔，默认值是0.5秒 &#x20;
4.  ignored\_exceptions：超时后的抛出的异常信息，默认抛出NoSuchElementExeception异常

```python
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# lambda
WebDriverWait(driver,timeout=10,poll_frequency=0.5).until(lambda diver: diver.find_element(By.ID, "kw"))
diver.find_element(By.ID, "kw").click()
# EC
WebDriverWait(driver,10,0.5).until(EC.presence_of_element_located((By.ID, "kw")))
diver.find_element(By.ID, "kw").click()

```





## 交互

### 浏览器基础操作

```python
# 打开网页
driver.get("https://www.baidu.com/")

# 浏览器最大化
driver.maximize_window()
# 设置浏览器宽、高
driver.set_window_size(600,600)

# 浏览器前进、后退、刷新
driver.forward()
driver.back()
driver.refresh()

# 获取标题
driver.title 
# 获取url
driver.current_url

```

### 弹窗消息

WebDriver提供了一个API，用于处理JavaScript提供的三种类型的原生弹窗消息（Alerts 警告框、Confirm 确认框、Prompt 提示框）。

#### 1.Alerts 警告框

Alters警告框，它显示一条自定义消息，以及一个用于关闭该警告的按钮。在大多数浏览器中标记为"确定"(OK)。

```python
driver.find_element(By.LINK_TEXT, "See an example alert").click()
alert = driver.switch_to.alert #切换到弹窗上
text = alert.text   # 获取文本
alert.accept()      # 接受(关闭)这些警告

```

#### 2.Confirm 确认框

Confirm确认框类似于警告框，不同之处在于用户还可以选择"取消"。

```python
driver.find_element(By.LINK_TEXT, "See a sample confirm").click()
alert = driver.switch_to.alert
text = alert.text
alert.dismiss()

```

#### 3.Prompt 提示框

Prompt提示框与确认框相似, 不同之处在于它们还包括文本输入

```python
driver.find_element(By.LINK_TEXT, "See a sample prompt").click()
alert = driver.switch_to.alert
text = alert.text
alert.send_keys("Selenium")
alert.accept()

```

### iframe

```html
<div id="modal">
  <iframe id="buttonframe"name="myframe"src="https://seleniumhq.github.io">
    <button>Click here</button>
  </iframe>
</div>
```

```python
# 通过 CSS_SELECTOR 切换框架
iframe = driver.find_element(By.CSS_SELECTOR, "#modal > iframe")
driver.switch_to.frame(iframe)
driver.find_element(By.TAG_NAME, 'button').click()

# 通过 id 切换框架
driver.switch_to.frame('buttonframe')
driver.find_element(By.TAG_NAME, 'button').click()

# 离开，切回到默认内容
driver.switch_to.default_content()

```

### 窗口

获取当前窗口句柄

```python
driver.current_window_handle

```

切换窗口/标签

```python
# 存储原始窗口的 ID
original_window = driver.current_window_handle
# 单击在新窗口中打开的链接
driver.find_element(By.LINK_TEXT, "new window").click()
# 等待新窗口或标签页
WebDriverWait(driver, 10).until(EC.number_of_windows_to_be(2))
# 循环执行，直到找到一个新的窗口句柄，切换
for window_handle in driver.window_handles:
    if window_handle != original_window:
        driver.switch_to.window(window_handle)
        break

```

创建新窗口/新标签页并且切换

```python
# 打开新标签页并切换到新标签页
driver.switch_to.new_window('tab')
# 打开一个新窗口并切换到新窗口
driver.switch_to.new_window('window')
```

关闭窗口或标签页

```python
# 关闭标签页或窗口
driver.close()
# 切回到之前的标签页或窗口
driver.switch_to.window(original_window)
```

退出浏览器

```python
driver.quit()
```



## Actions接口

用于向 Web 浏览器提供虚拟化设备输入操作的低级接口。分别有键盘、鼠标、笔 、滚轮。

先引入基础的Actions相关包

```python
from selenium.webdriver import ActionChains
from selenium.webdriver.common.actions.action_builder import ActionBuilder

```

暂停

```python
clickable = driver.find_element(By.ID, "clickable")
ActionChains(driver)\
      .move_to_element(clickable)\
      .pause(1)\
      .click_and_hold()\
      .pause(1)\
      .send_keys("abc")\
      .perform()
```

释放所有Actions，驱动程序会记住整个会话中所有输入项的状态. 即使创建actions类的新实例, 按下的键和指针的位置 也将处于以前执行的操作离开它们的任何状态，需要释放所有当前按下的键和指针按钮。

```python
ActionBuilder(driver).clear_actions()
```

### 键盘

按下按键: `key_down(Keys.SHIFT)`

释放按键：`key_up(Keys.SHIFT)`

键入：`send_keys_to_elemeny(element,'a')`

指定元素键入：`send_keys('a')`

tips：使用send\_keys\_to\_element后会覆盖元素内原来的文本，send\_keys是追加

```python
from selenium.webdriver import Keys


# 输入'A'
# 操作流程: shift, 输入'a', 松开shift
driver.find_element(By.ID, "textInput").send_keys(Keys.SHIFT,"a")

# 输入'Ab'
# 操作流程: shift, 输入'a', 松开shift, 输入'bb'
text = driver.find_element(By.ID, "textInput")
ActionChains(driver) \
    .key_down(Keys.SHIFT) \
    .send_keys_to_element(text,"a") \
    .key_up(Keys.SHIFT) \
    .send_keys("b") \
    .perform()

# 复制粘贴，文本框内显示'gptgpt'
# 操作流程: 输入'gpt', command+a, command+a, 方向键右, command+v
ActionChains(driver) \
    .send_keys_to_element(text,"gpt") \
    .key_down(Keys.COMMAND) \
    .send_keys("ac") \
    .key_up(Keys.COMMAND) \
    .send_keys(Keys.ARROW_RIGHT)\
    .key_down(Keys.COMMAND) \
    .send_keys("v") \
    .key_up(Keys.COMMAND) \
    .perform()

```

### 鼠标

鼠标操作只有三种：按下、释放、移动

```python
from selenium.webdriver.common.actions.mouse_button import MouseButton


# 单击并按住 click_and_hold
el = driver.find_element(By.ID, "clickable")
ActionChains(driver) \
    .click_and_hold(el) \
    .perform()

# 单击 click 
el = driver.find_element(By.ID, "click")
ActionChains(driver)\
    .click(el)\
    .perform()
    
# 双击 double_click
el = driver.find_element(By.ID, "click")
ActionChains(driver)\
    .double_click(el)\
    .perform()

# 左键单击 context_click
el = driver.find_element(By.ID, "clickable")
ActionChains(driver) \
    .context_click(el) \
    .perform()

# 后退点击（目前没有便捷API）
action = ActionBuilder(driver)
action.pointer_action.pointer_down(MouseButton.BACK)
action.pointer_action.pointer_up(MouseButton.BACK)
action.perform()

# 前进点击（目前没有便捷API）
action = ActionBuilder(driver)
action.pointer_action.pointer_down(MouseButton.FORWARD)
action.pointer_action.pointer_up(MouseButton.FORWARD)
action.perform()

# 悬停 move_to_element
el = driver.find_element(By.ID, "hover")
ActionChains(driver)\
        .move_to_element(el)\
        .perform()

# 从元素偏移 第一个数值参数为向右，第二个数值参数为向下       
el= driver.find_element(By.ID, "mouse-tracker")
ActionChains(driver) \
    .move_to_element_with_offset(el, 8, 0) \
    .perform()
# 从视口偏移（从视口的左上角开始）
action = ActionBuilder(driver)
action.pointer_action.move_to_location(8, 0)
action.perform()
# 从当前指针位置偏移（默认为视口的左上角）
ActionChains(driver)\
    .move_by_offset(13,15)\
    .perform()
    
# 拖放
draggable = driver.find_element(By.ID, "draggable")
droppable = driver.find_element(By.ID, "droppable")
ActionChains(driver) \
    .drag_and_drop(draggable, droppable) \
    .perform()
# 按偏移量拖放
draggable = driver.find_element(By.ID, "draggable")
start = driver.find_element(By.ID, "draggable").location
finish = driver.find_element(By.ID, "droppable").location
ActionChains(driver)\
        .drag_and_drop_by_offset(draggable, finish['x'] - start['x'], finish['y'] - start['y'])\
        .perform()

```

### 笔

```python
from selenium.webdriver.common.actions.interaction import POINTER_PEN
from selenium.webdriver.common.actions.pointer_input import PointerInput

pointer_area = driver.find_element(By.ID, "pointerArea")
pen_input = PointerInput(POINTER_PEN, "default pen")
action = ActionBuilder(driver, mouse=pen_input)
action.pointer_action \
    .move_to(pointer_area) \
    .pointer_down() \
    .move_by(2, 2) \
    .pointer_up()
action.perform()

pointer_area = driver.find_element(By.ID, "pointerArea")
pen_input = PointerInput(POINTER_PEN, "default pen")
action = ActionBuilder(driver, mouse=pen_input)
action.pointer_action \
    .move_to(pointer_area) \
    .pointer_down() \
    .move_by(2, 2, tilt_x=-72, tilt_y=9, twist=86) \
    .pointer_up(0)
action.perform()

```

### 滚轮

```python
from selenium.webdriver.common.actions.wheel_input import ScrollOrigin

# 滚动到元素
# 滚动后元素的底部位于屏幕底部
iframe = driver.find_element(By.TAG_NAME, "iframe")
ActionChains(driver) \
    .scroll_to_element(iframe) \
    .perform()

# 定量滑动
# scroll_by_amount 第一个参数为delta_x表示向右滚动量,第二个参数为delta_y表示向下滚动量
ActionChains(driver) \
    .scroll_by_amount(0, 200) \
    .perform()

# 从元素开始定量滚动
# ScrollOrigin.from_element() 如果元素后不加其他参数，滚动原点是元素的中心或视口的左上角
# scroll_from_origin()有三个参数 第一个表示起始点元素，第二个是 delta_x，第三个是 delta_y
# 起始点不在视口中，则起始点元素的底部将首先滚动到视口的底部，再滚动。
iframe = driver.find_element(By.TAG_NAME, "iframe")
scroll_origin = ScrollOrigin.from_element(iframe)
ActionChains(driver) \
    .scroll_from_origin(scroll_origin, 0, 200) \
    .perform()
    
footer = driver.find_element(By.TAG_NAME, "footer")
scroll_origin = ScrollOrigin.from_element(footer, 0, -50)
ActionChains(driver) \
    .scroll_from_origin(scroll_origin, 0, 200) \
    .perform()
    
# 从视口开始定量滚动
# 特别注意：如果视口左上角的偏移量落在屏幕之外，则会导致异常
scroll_origin = ScrollOrigin.from_viewport(200, 200)
ActionChains(driver) \
    .scroll_from_origin(scroll_origin, 0, 2000) \
    .perform()

```

