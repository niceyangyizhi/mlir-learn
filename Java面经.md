1. 谈谈JVM调优

大多数情况下不需要做JVM调优，因为JVM的默认参数已经是JVM团队反复测试后的给出推荐参数。不过，为了保证核心服务的可靠性，可以给核心服务的一些重要JVM指标添加监控告警，当出现异常波动时可以人为介入分析评估。
JVM重要指标：
- jvm.gc.time
- jvm.gc.meantime
- jvm.fullgc.count
- jvm.fullgc.time

当发现性能指标出现问题后，可以按照如下思路进行排查分析：
- 定位当前系统的瓶颈
- 确定优化目标
- 制订优化方案
- 对比优化前后的指标，统计优化效果
- 持续观察和追踪优化效果


Java基础语法
Java容器
# Java多线程
线程池
Java虚拟机

SpringBoot


MYSQL

