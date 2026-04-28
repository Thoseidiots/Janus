# Verification Script for Installed Tools
# Run this after restarting terminal

Write-Output @"
╔══════════════════════════════════════════════════════════════════════════╗
║                    INSTALLATION VERIFICATION                             ║
╚══════════════════════════════════════════════════════════════════════════╝

"@

$results = @()

# Check Node.js
Write-Output "[1/11] Checking Node.js..."
try {
    $version = node --version 2>$null
    if ($version) {
        Write-Output "  ✓ Node.js: $version"
        $results += @{Tool="Node.js"; Status="✓"; Version=$version}
    }
} catch {
    Write-Output "  ✗ Node.js: Not found"
    $results += @{Tool="Node.js"; Status="✗"; Version="Not found"}
}

# Check npm
Write-Output "[2/11] Checking npm..."
try {
    $version = npm --version 2>$null
    if ($version) {
        Write-Output "  ✓ npm: v$version"
        $results += @{Tool="npm"; Status="✓"; Version="v$version"}
    }
} catch {
    Write-Output "  ✗ npm: Not found"
    $results += @{Tool="npm"; Status="✗"; Version="Not found"}
}

# Check yarn
Write-Output "[3/11] Checking yarn..."
try {
    $version = yarn --version 2>$null
    if ($version) {
        Write-Output "  ✓ yarn: v$version"
        $results += @{Tool="yarn"; Status="✓"; Version="v$version"}
    }
} catch {
    Write-Output "  ✗ yarn: Not found"
    $results += @{Tool="yarn"; Status="✗"; Version="Not found"}
}

# Check pnpm
Write-Output "[4/11] Checking pnpm..."
try {
    $version = pnpm --version 2>$null
    if ($version) {
        Write-Output "  ✓ pnpm: v$version"
        $results += @{Tool="pnpm"; Status="✓"; Version="v$version"}
    }
} catch {
    Write-Output "  ✗ pnpm: Not found"
    $results += @{Tool="pnpm"; Status="✗"; Version="Not found"}
}

# Check TypeScript
Write-Output "[5/11] Checking TypeScript..."
try {
    $version = tsc --version 2>$null
    if ($version) {
        Write-Output "  ✓ TypeScript: $version"
        $results += @{Tool="TypeScript"; Status="✓"; Version=$version}
    }
} catch {
    Write-Output "  ✗ TypeScript: Not found"
    $results += @{Tool="TypeScript"; Status="✗"; Version="Not found"}
}

# Check ts-node
Write-Output "[6/11] Checking ts-node..."
try {
    $version = ts-node --version 2>$null
    if ($version) {
        Write-Output "  ✓ ts-node: $version"
        $results += @{Tool="ts-node"; Status="✓"; Version=$version}
    }
} catch {
    Write-Output "  ✗ ts-node: Not found"
    $results += @{Tool="ts-node"; Status="✗"; Version="Not found"}
}

# Check jest
Write-Output "[7/11] Checking jest..."
try {
    $version = jest --version 2>$null
    if ($version) {
        Write-Output "  ✓ jest: v$version"
        $results += @{Tool="jest"; Status="✓"; Version="v$version"}
    }
} catch {
    Write-Output "  ✗ jest: Not found"
    $results += @{Tool="jest"; Status="✗"; Version="Not found"}
}

# Check eslint
Write-Output "[8/11] Checking eslint..."
try {
    $version = eslint --version 2>$null
    if ($version) {
        Write-Output "  ✓ eslint: $version"
        $results += @{Tool="eslint"; Status="✓"; Version=$version}
    }
} catch {
    Write-Output "  ✗ eslint: Not found"
    $results += @{Tool="eslint"; Status="✗"; Version="Not found"}
}

# Check prettier
Write-Output "[9/11] Checking prettier..."
try {
    $version = prettier --version 2>$null
    if ($version) {
        Write-Output "  ✓ prettier: v$version"
        $results += @{Tool="prettier"; Status="✓"; Version="v$version"}
    }
} catch {
    Write-Output "  ✗ prettier: Not found"
    $results += @{Tool="prettier"; Status="✗"; Version="Not found"}
}

# Check PostgreSQL
Write-Output "[10/11] Checking PostgreSQL..."
try {
    $version = psql --version 2>$null
    if ($version) {
        Write-Output "  ✓ PostgreSQL: $version"
        $results += @{Tool="PostgreSQL"; Status="✓"; Version=$version}
    }
} catch {
    Write-Output "  ✗ PostgreSQL: Not found"
    $results += @{Tool="PostgreSQL"; Status="✗"; Version="Not found"}
}

# Check CMake
Write-Output "[11/11] Checking CMake..."
try {
    $version = cmake --version 2>$null | Select-Object -First 1
    if ($version) {
        Write-Output "  ✓ CMake: $version"
        $results += @{Tool="CMake"; Status="✓"; Version=$version}
    }
} catch {
    Write-Output "  ✗ CMake: Not found"
    $results += @{Tool="CMake"; Status="✗"; Version="Not found"}
}

# Summary
Write-Output ""
Write-Output "="*70
Write-Output "SUMMARY"
Write-Output "="*70

$installed = ($results | Where-Object { $_.Status -eq "✓" }).Count
$missing = ($results | Where-Object { $_.Status -eq "✗" }).Count

Write-Output "Installed: $installed / $($results.Count)"
Write-Output "Missing: $missing / $($results.Count)"

if ($missing -gt 0) {
    Write-Output ""
    Write-Output "Missing tools:"
    $results | Where-Object { $_.Status -eq "✗" } | ForEach-Object {
        Write-Output "  - $($_.Tool)"
    }
}

Write-Output ""
Write-Output "="*70
