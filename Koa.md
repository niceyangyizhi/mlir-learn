# 简介

koa是基于Node.js平台的下一代web开发框架。基于koa的web应用是一个包含一系列中间价的对象。这些中间件以类似洋葱的结构组织，依次对request请求进行处理。Koa的hello world示例如下：

```
const Koa = require('koa');
// 创建一个Koa应用
const app = new Koa();

// 在应用中添加一个中间件
app.use(async ctx => {
  ctx.body = 'Hello World';
});

// 在3000端口监听request请求
app.listen(3000);
```
# 中间件级联

```
const Koa = require('koa');
const app = new Koa();

// x-response-time

app.use(async (ctx, next) => {
  const start = Date.now();
  await next();
  const ms = Date.now() - start;
  ctx.set('X-Response-Time', `${ms}ms`);
});

// logger

app.use(async (ctx, next) => {
  const start = Date.now();
  await next();
  const ms = Date.now() - start;
  console.log(`${ctx.method} ${ctx.url} - ${ms}`);
});

// response

app.use(async ctx => {
  ctx.body = 'Hello World';
});

app.listen(3000);
```

上面的例子中，请求会依次经过x-response-time、logging、response中间件。当一个中间件调用`next()`函数的时候，函数会挂起并将控制权传递给下一个中间件。在没有更下游的中间件时，将会从堆栈中依次弹出中间件，以执行起上游行为。


# Context(上下文)

Koa将node的request和response对象封装在一个单独的对象context里面。每个request请求都会创建一个context对象，context在中间件函数中作为函数参数来引用。

```
app.use(async ctx => {
  ctx; // is the Context
  ctx.request; // is a koa Request
  ctx.response; // is a koa Response
});
```

## API 

### ctx.req

Node 的 request 对象。

### ctx.res

Node 的 response 对象。

Koa 不支持 直接调用底层 res 进行响应处理。请避免使用以下 node 属性:

- res.statusCode
- res.writeHead()
- res.write()
- res.end()

### ctx.request

Koa 的 Request 对象。

### ctx.response

Koa 的 Response 对象。

### ctx.state

推荐的命名空间，用于通过中间件传递信息到前端视图

```
ctx.state.user = await User.find(id);
```

### ctx.app

应用实例引用。

### ctx.cookies.get(name, [options])

获得 cookie 中名为 name 的值，options 为可选参数

### ctx.cookies.set(name, value, [options])

设置 cookie 中名为 name 的值，options 为可选参数: 

- maxAge 一个数字，表示 Date.now()到期的毫秒数
- signed 是否要做签名
- expires cookie有效期
- pathcookie 的路径，默认为 /'
- domain cookie 的域
- secure false 表示 cookie 通过 HTTP 协议发送，true 表示 cookie 通过 HTTPS 发送。
- httpOnly true 表示 cookie 只能通过 HTTP 协议发送
- overwrite 一个布尔值，表示是否覆盖以前设置的同名的Cookie（默认为false）。 如果为true，在设置此cookie时，将在同一请求中使用相同名称（不管路径或域）设置的所有Cookie将从Set-Cookie头部中过滤掉。

### ctx.throw([status], [msg], [properties])

抛出包含 .status 属性的错误，默认为 500。

### ctx.assert(value, [status], [msg], [properties])

当!value时， 方法抛出一个类似.throw()的错误.

