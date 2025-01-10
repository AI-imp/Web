package main

import (
	"fmt"
	"net/http"
)

// 处理首页请求
func homeHandler(w http.ResponseWriter, r *http.Request) {
	// 设置响应头
	w.Header().Set("Content-Type", "text/html")
	// 返回简单的 HTML
	fmt.Fprintf(w, "<html><body><h1>欢迎来到我的简单网站!</h1><p>这是一个用Go构建的简单网站。</p></body></html>")
}

func main() {
	// 路由
	http.HandleFunc("/", homeHandler)

	// 启动 Web 服务器，监听本地 8080 端口
	fmt.Println("服务器启动，访问 http://localhost:8080")
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		fmt.Println("错误启动服务器:", err)
	}
}
