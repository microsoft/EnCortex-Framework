# Errors and Fixes

## Docker

1. Error: `error getting credentials - err: exit status 1, out: GDBus.Error:org.freedesktop.DBus.Error.ServiceUnknown: The name org.freedesktop.secrets was not provided by any .service files`

 - Fix: `sudo apt install gnupg2 pass`