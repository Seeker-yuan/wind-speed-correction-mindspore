# GitHub上传步骤

## 📋 完整步骤

### 第一步：在GitHub上创建仓库

1. 打开浏览器，访问 https://github.com
2. 登录你的GitHub账号
3. 点击右上角的 "+" 号，选择 "New repository"
4. 填写仓库信息：
   - Repository name: `wind-speed-correction-mindspore`（或你喜欢的名字）
   - Description: `风能发电机风速误差校正系统 - 基于华为MindSpore`
   - 选择 Public（公开）或 Private（私有）
   - **不要勾选** "Initialize this repository with a README"
5. 点击 "Create repository"

### 第二步：配置Git（如果首次使用）

在PowerShell或Git Bash中运行：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"
```

### 第三步：初始化本地仓库并上传

在 PowerShell 中，进入项目目录并执行：

```powershell
# 1. 进入项目目录
cd "C:\Users\31876\Desktop\风能ui设计"

# 2. 初始化Git仓库
git init

# 3. 添加所有文件（.gitignore会自动排除不需要的文件）
git add .

# 4. 创建第一次提交
git commit -m "首次提交：基于MindSpore的风速误差校正系统"

# 5. 添加远程仓库（替换成你的仓库地址）
git remote add origin https://github.com/你的用户名/wind-speed-correction-mindspore.git

# 6. 推送到GitHub
git push -u origin master
```

**注意**：第5步的仓库地址需要替换成你在第一步创建的仓库地址

### 第四步：验证

1. 刷新你的GitHub仓库页面
2. 应该能看到所有代码文件已上传
3. README.md会自动显示在仓库首页

---

## 🔧 如果遇到问题

### 问题1：git命令不存在

**解决方案**：安装Git

1. 下载：https://git-scm.com/download/win
2. 安装（一路下一步）
3. 重新打开PowerShell

### 问题2：推送时要求输入用户名密码

**解决方案**：使用Personal Access Token

1. GitHub设置 → Developer settings → Personal access tokens → Generate new token
2. 勾选 `repo` 权限
3. 生成token并保存
4. 推送时：
   - Username: 你的GitHub用户名
   - Password: 粘贴刚才的token（不是密码）

### 问题3：推送被拒绝

**解决方案**：

```bash
# 如果GitHub上有文件，先拉取
git pull origin master --allow-unrelated-histories

# 然后再推送
git push -u origin master
```

---

## 📝 后续更新代码

以后修改代码后，使用以下命令更新：

```bash
# 1. 查看修改了哪些文件
git status

# 2. 添加修改的文件
git add .

# 3. 提交修改
git commit -m "描述你的修改内容"

# 4. 推送到GitHub
git push
```

---

## 🎯 快速命令参考

```bash
# 查看状态
git status

# 查看提交历史
git log --oneline

# 撤销修改
git checkout -- 文件名

# 查看远程仓库
git remote -v

# 拉取最新代码
git pull
```

---

## ✅ 检查清单

上传前确认：
- [ ] 已在GitHub创建仓库
- [ ] 已配置Git用户名和邮箱
- [ ] 已安装Git
- [ ] 已准备好GitHub用户名和token
- [ ] 项目路径正确
- [ ] .gitignore文件已创建（自动排除数据文件）

---

**准备好了就可以开始上传了！** 🚀
