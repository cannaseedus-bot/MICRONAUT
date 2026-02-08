# MICRONAUT ORCHESTRATOR (SCO/1 projection only)

$Root = Split-Path $MyInvocation.MyCommand.Path
$IO = Join-Path $Root "io"
$Chat = Join-Path $IO "chat.txt"
$Stream = Join-Path $IO "stream.txt"
$Object = Join-Path $Root "micronaut.s7"

Write-Host "Micronaut online."

function Invoke-MicronautMuOp {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Op,
        [Parameter(Mandatory = $true)]
        [string]$Payload
    )

    if (-not (Test-Path $Object)) {
        throw "Micronaut object not found at $Object"
    }

    $output = & $Object $Op $Payload
    return $output
}

function cm1_verify {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Entry
    )

    $result = Invoke-MicronautMuOp -Op "cm1-verify" -Payload $Entry
    return $result -eq "ok"
}

function Invoke-KUHUL-TSG {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Input
    )

    return Invoke-MicronautMuOp -Op "kuhul-tsg" -Payload $Input
}

function Invoke-SCXQ2-Infer {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Signal
    )

    return Invoke-MicronautMuOp -Op "scxq2-infer" -Payload $Signal
}

$lastSize = 0

while ($true) {
    if (Test-Path $Chat) {
        $size = (Get-Item $Chat).Length
        if ($size -gt $lastSize) {
            $entry = Get-Content $Chat -Raw
            $lastSize = $size

            if (-not (cm1_verify $entry)) {
                Write-Host "CM-1 violation"
                continue
            }

            $signal = Invoke-KUHUL-TSG -Input $entry
            $response = Invoke-SCXQ2-Infer -Signal $signal

            Add-Content $Stream ">> $response"
        }
    }
    Start-Sleep -Milliseconds 200
}
