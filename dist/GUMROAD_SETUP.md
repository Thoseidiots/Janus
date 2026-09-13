# Gumroad Product Setup — Universal Oxpecker v1.2.0

**Goal:** Create professional product listing on Gumroad with $0.99/month licensing model + free pass option

---

## Step 1: Go to Gumroad & Create Product

1. Visit: https://gumroad.com/
2. Sign in (or create account)
3. Click **"Create a product"**
4. Fill in:
   - **Product Name:** `Universal Oxpecker v1.2.0`
   - **Product Type:** Software
   - **Category:** Developer Tools / Debugging

---

## Step 2: Product Description

Copy & paste this into the description field:

```
Universal Oxpecker — Multi-language Debugger, Scanner & Auto-Repair

Find and fix bugs across 12+ programming languages with a single tool.

✨ FEATURES
• Multi-language analysis (Python, JavaScript/TypeScript, Java, C/C++, Rust, Go, C#, Ruby, PHP, Lua, Swift, Zig)
• Smart bug detection with severity levels (Error, Warning, Info)
• Stronger-safe auto-repair (fixes low-risk issues automatically)
• Medium & high-risk candidates held for your review
• Full rollback history — undo any repair at any time
• Zero cloud required — works completely offline
• 8+ parallel workers for fast project-wide scans

🔧 QUICK START
After installation:

$ oxpecker license activate --key YOUR_KEY --email you@example.com
$ oxpecker scan ./my-project          # Find bugs
$ oxpecker repair ./buggy-file.py     # Auto-fix

View all commands:
$ oxpecker --help

📦 INSTALLATION
Requires Python 3.8+

Install via pip:
$ pip install universal-oxpecker

Or from source:
$ git clone https://github.com/thoseidiots/janus
$ cd janus
$ pip install -e tools/universal_oxpecker

💰 PRICING
• $0.99/month base price
• Optional donations: $5, $10, $20+ (appreciated but not required)
• Cannot afford it? Click "I cannot afford this" for a free 30-day trial
• No guilt, no judgment — everyone's welcome

📄 LICENSE
Monthly subscription. Auto-renews on your anniversary date unless cancelled.
Free trial via "cannot afford" button always available.

🤝 SUPPORT
• GitHub: https://github.com/thoseidiots/janus/issues
• Documentation: Included with product

---

Made with ♥ for developers who want their code to be better.
```

---

## Step 3: Set Pricing

1. Click **"Pricing"** section
2. Set **Base Price:** $0.99
3. Enable **"Pay What You Want"**
   - Allow users to pay more than $0.99 (for donations)
   - Check: "Let customers name their own price"
   - Minimum: $0.99
4. Enable **"I cannot afford this"**
   - Check the option
   - Message: "No problem! You can use Oxpecker for free this month. Activate with the license key below."

---

## Step 4: License Key Setup

1. Click **"License Keys"** or **"Licensing"** (varies by Gumroad version)
2. Enable product licensing
3. **License Key Format:** (leave blank or use UUID if prompted)
4. Copy the **Product ID** and **Product Token** — you'll need these for API validation

Save these as environment variables for license validation:
```bash
export OXPECKER_PRODUCT_ID="your-product-id-here"
export OXPECKER_PRODUCT_TOKEN="your-product-token-here"
```

---

## Step 5: Files & Downloads

### Main File (Required)
- **Name:** `Universal Oxpecker Installation Guide`
- **File:** Upload `dist/LAUNCH_READY.md` or create new file with installation instructions

### Installation Instructions (Copy-Paste)

Create a text file called `INSTALL.txt`:
```
UNIVERSAL OXPECKER v1.2.0 — INSTALLATION GUIDE

🎯 SYSTEM REQUIREMENTS
• Python 3.8 or higher
• Windows, macOS, or Linux

📥 INSTALLATION STEPS

1. Check Python version:
   $ python --version
   (Should be 3.8.0 or higher)

2. Install Oxpecker:
   $ pip install universal-oxpecker

3. Activate your license:
   $ oxpecker license activate --key YOUR_LICENSE_KEY --email your@email.com

4. Test installation:
   $ oxpecker version

5. Scan your first project:
   $ oxpecker scan ~/my-project

🎓 COMMON COMMANDS

Show all commands:
  $ oxpecker --help

List supported languages:
  $ oxpecker langs

Deep analysis on single file:
  $ oxpecker debug app.py
  $ oxpecker check app.js

Scan entire project:
  $ oxpecker scan ./src --workers 8

Auto-repair low-risk issues:
  $ oxpecker repair app.py
  $ oxpecker repair ./src --project --max-rounds 5

View repair candidates (medium-risk):
  $ oxpecker repair app.py --show-candidates 10

Undo repairs:
  $ oxpecker rollback ./oxpecker-working-copies/app.py --steps 1

Manage license:
  $ oxpecker license status
  $ oxpecker license cannot-afford      # Free 30-day trial
  $ oxpecker license renew              # Extend subscription
  $ oxpecker license remove             # Uninstall

💰 LICENSE REMINDER
• Your $0.99/month license auto-renews on the anniversary of activation
• Activate "I cannot afford" for a free 30-day pass
• Donate extra if you love it!

❓ NEED HELP?
Visit: https://github.com/thoseidiots/janus/issues
Or email: support@oxpecker.dev

Happy debugging! 🐛🔧
```

---

## Step 6: Product Preview & Publish

1. Click **"Preview"** to see how it looks
2. Review:
   - Title is clear
   - Description is compelling
   - Price shows $0.99 minimum
   - "I cannot afford" option is visible
   - Installation guide is downloadable
3. Click **"Publish"** 🚀

---

## Step 7: Get License Credentials

After publishing:

1. Go to product settings
2. Find **"License Keys"** section
3. Copy:
   - **Product ID** (looks like a UUID)
   - **Product Token** (secret key)
4. Save these securely

**Next:** Update license_manager.py with these credentials for Gumroad API validation

---

## Step 8: Verify Gumroad Integration

Test the Gumroad flow:

1. As a buyer (incognito window):
   - Visit your Gumroad product page
   - Click "Pay $0.99"
   - Complete payment
   - Receive license key
   - Activate: `oxpecker license activate --key KEY --email me@example.com`

2. Test "I cannot afford":
   - Click "I cannot afford this"
   - Receive free trial key
   - Verify it works for 30 days

---

## Marketing Next Steps

Once product is live on Gumroad:

1. **Create GitHub Release** (v1.2.0)
   - Add Gumroad link
   - Include download instructions

2. **Announce on Social**
   - Twitter/X: "Universal Oxpecker is live! Find bugs across 12+ languages for $0.99/month. Free trial available."
   - Reddit: r/programming, r/python, r/rust, language-specific subreddits
   - Hacker News (if appropriate)

3. **Monitor**
   - Track first downloads
   - Respond to user questions
   - Fix any critical bugs same-day

---

## Expected Timeline

- **Setup:** 15-20 minutes
- **Gumroad Review:** Usually instant, sometimes up to 1 hour
- **First customers:** Within 24-48 hours of announcement
- **Steady state:** 5-10 customers/week (scaling over time)

---

## Gumroad Product Link Template

Once created, your product will be at:
```
https://gumroad.com/@YOUR_USERNAME/oxpecker
```

Share this link everywhere!

---

**Ready? Go set it up!** 🚀
