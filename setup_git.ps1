# PowerShell script to set up git repository and push to GitHub
# Run this script after installing Git

Write-Host "Setting up Git repository..." -ForegroundColor Green

# Check if git is available
try {
    git --version
    Write-Host "Git is available" -ForegroundColor Green
} catch {
    Write-Host "Git is not installed. Please install Git first from https://git-scm.com/" -ForegroundColor Red
    exit 1
}

# Initialize git repository
Write-Host "Initializing git repository..." -ForegroundColor Yellow
git init

# Add all files
Write-Host "Adding files to git..." -ForegroundColor Yellow
git add .

# Create initial commit
Write-Host "Creating commit..." -ForegroundColor Yellow
git commit -m "simple setup and backward configuration"

# Instructions for adding remote and pushing
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Add your GitHub repository as remote:" -ForegroundColor White
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Push to GitHub:" -ForegroundColor White
Write-Host "   git branch -M main" -ForegroundColor Gray
Write-Host "   git push -u origin main" -ForegroundColor Gray
Write-Host ""
Write-Host "Replace YOUR_USERNAME and YOUR_REPO_NAME with your actual GitHub details" -ForegroundColor Yellow
