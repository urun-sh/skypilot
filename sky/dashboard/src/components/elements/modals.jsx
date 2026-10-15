import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { CopyIcon } from 'lucide-react';
import { CustomTooltip as Tooltip } from '@/components/utils';
import { BASE_PATH } from '@/data/connectors/constants';
import { useMobile } from '@/hooks/useMobile';

export function SSHInstructionsModal({ isOpen, onClose, cluster }) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // urun: this dashboard fronts a CONTROLLER-managed fleet — the user's
  // machine has no ~/.sky state for these clusters, so upstream's
  // `ssh <cluster>` hint cannot work. The image ships /usr/local/bin/
  // urun-ssh, which resolves the per-cluster key + User/HostName from
  // sky's own generated config under the runner HOME, so the command stays
  // SHORT (a long inline ssh invocation broke on terminal line-wrap — live
  // report 2026-10-14) and gets a real interactive TTY via -it.
  // Pin the kubectl context to THIS dashboard's region: the user's default
  // context is often a dev/other cluster where the prod fleet's keys do not
  // exist (live report 2026-10-14: default dev-usw2, dashboard on usw2).
  // The dashboard hostname carries our region id
  // (skydashboard.<region>.prod.priv.urun.sh) — exactly the `<region>` in
  // the `prod-<region>` context name.
  const regionMatch =
    typeof window !== 'undefined'
      ? window.location.hostname.match(
          /\b(usw2|use2|usw1|use1|usw3|use3|usc1|cac1|euw1|apse1)\b/
        )
      : null;
  const kubeContext = regionMatch ? ` --context prod-${regionMatch[1]}` : '';
  const sshCommand =
    `kubectl${kubeContext} -n urun exec -it deploy/skypilot-controller ` +
    `-c manager -- urun-ssh ${cluster}`;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            Connect to: <span className="font-light">{cluster}</span>
          </DialogTitle>
          <DialogDescription>
            SSH into this cluster from your machine (kubectl exec into the
            controller, which holds the fleet&apos;s per-cluster credentials).
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col space-y-4">
          <div>
            <h3 className="text-sm font-medium mb-2">SSH Command</h3>
            <Card className="p-3 bg-gray-50">
              <div className="flex items-center justify-between">
                <pre className="text-sm w-full whitespace-pre-wrap break-all">
                  <code className="block">{sshCommand}</code>
                </pre>
                <Tooltip content={copied ? 'Copied!' : 'Copy command'}>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => handleCopy(sshCommand)}
                    className="h-8 w-8 rounded-full"
                  >
                    <CopyIcon className="h-4 w-4" />
                  </Button>
                </Tooltip>
              </div>
            </Card>
          </div>

          <div>
            <h3 className="text-sm font-medium mb-2">Additional Information</h3>
            <p className="text-sm text-secondary-foreground">
              Requires kubectl access to the prod cluster. The command opens an
              interactive shell through the skypilot-controller manager
              container, which holds the per-cluster key and resolves the
              cluster&apos;s address for you.
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function VSCodeInstructionsModal({ isOpen, onClose, cluster }) {
  const isMobile = useMobile();

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>
            Connect to: <span className="font-light">{cluster}</span>
          </DialogTitle>
          <DialogDescription>
            <div className="flex flex-col space-y-4">
              <div>
                <h3 className="text-sm font-medium mb-2 my-2">
                  Setup SSH access
                </h3>
                <Card className="p-3 bg-gray-50">
                  <div className="flex items-center justify-between">
                    <pre className="text-sm">
                      <code>sky status {cluster}</code>
                    </pre>
                    <Tooltip content="Copy command">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() =>
                          navigator.clipboard.writeText(`sky status ${cluster}`)
                        }
                        className="h-8 w-8 rounded-full"
                      >
                        <CopyIcon className="h-4 w-4" />
                      </Button>
                    </Tooltip>
                  </div>
                </Card>
              </div>
              <div>
                <h3 className="text-sm font-medium mb-2 my-2">
                  Connect with VSCode/Cursor
                </h3>
                <Card className="p-3 bg-gray-50">
                  <div className="flex items-center justify-between">
                    <pre className="text-sm">
                      <code>
                        code --remote ssh-remote+{cluster} &quot;/home&quot;
                      </code>
                    </pre>
                    <Tooltip content="Copy command">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() =>
                          navigator.clipboard.writeText(
                            `code --remote ssh-remote+${cluster} "/home"`
                          )
                        }
                        className="h-8 w-8 rounded-full"
                      >
                        <CopyIcon className="h-4 w-4" />
                      </Button>
                    </Tooltip>
                  </div>
                </Card>
              </div>
              <div>
                <h3 className="text-sm font-medium">
                  Or use the GUI to connect
                </h3>
                <div
                  className={`relative ${isMobile ? '-mt-5' : '-mt-10'}`}
                  style={{ paddingBottom: '70%' }}
                >
                  <video
                    className="absolute top-0 left-0 w-full h-full rounded-lg"
                    controls
                    autoPlay
                    muted
                    preload="metadata"
                  >
                    <source
                      src={`${BASE_PATH}/videos/cursor-small.mp4`}
                      type="video/mp4"
                    />
                    Your browser does not support the video tag.
                  </video>
                </div>
              </div>
            </div>
          </DialogDescription>
        </DialogHeader>
      </DialogContent>
    </Dialog>
  );
}

export function ConfirmationModal({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = 'Confirm',
  confirmVariant = 'destructive',
  confirmClassName = null,
}) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{message}</DialogDescription>
        </DialogHeader>
        <DialogFooter className="flex justify-end gap-2 pt-4">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant={confirmClassName ? undefined : confirmVariant}
            className={confirmClassName}
            onClick={() => {
              onConfirm();
              onClose();
            }}
          >
            {confirmText}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
