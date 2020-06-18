# A cheetsheet for jekyll and MM theme.

Text alignment:

Left aligned text
{: .text-left}

Right aligned text.
{: .text-right}

Image alignmentPermalink

![image-center](/assets/images/filename.jpg){: .align-center}

![image-left](/assets/images/filename.jpg){: .align-left}

![image-right](/assets/images/filename.jpg){: .align-right}

![full](/assets/images/filename.jpg)
{: .full}



Buttons
<a href="#" class="btn btn--primary">Link Text</a>
Button          Type             Example Class                      Kramdown

Default         Text            .btn                                [Text](#link){: .btn}
Primary         Text            .btn .btn--primary                  [Text](#link){: .btn .btn--primary}
Success         Text            .btn .btn--success                  [Text](#link){: .btn .btn--success}
Warning         Text            .btn .btn--warning                  [Text](#link){: .btn .btn--warning}
Danger          Text            .btn .btn--danger                   [Text](#link){: .btn .btn--danger}
Info            Text            .btn .btn--info                     [Text](#link){: .btn .btn--info}
Inverse         Text            .btn .btn--inverse                  [Text](#link){: .btn .btn--inverse}
Light Outline   Text            .btn .btn--light-outline            [Text](#link){: .btn .btn--light-outline}


Watch out! This paragraph of text has been emphasized with the {: .notice} class.

Watch out! This paragraph of text has been emphasized with the {: .notice--primary} class.

Watch out! This paragraph of text has been emphasized with the {: .notice--info} class.

Watch out! This paragraph of text has been emphasized with the {: .notice--warning} class.

Watch out! This paragraph of text has been emphasized with the {: .notice--success} class.

Watch out! This paragraph of text has been emphasized with the {: .notice--danger} class.
